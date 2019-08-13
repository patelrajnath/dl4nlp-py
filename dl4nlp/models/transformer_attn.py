#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
 Copyright (c) 2018 Raj Nath Patel
 Licensed under the GNU Public License.
 Author: Raj Nath Patel
 Email: patelrajnath (at) gmail (dot) com
 Created: 25/May/2018 03:16
 """
import torch
import copy


from torch.nn import functional as F
from torch import nn

from dl4nlp.models import register_model, register_model_architecture
from dl4nlp.models.dl4nlp_model import BaseModel
from dl4nlp.models.model_attenntion import MultiHeadedAttention
from dl4nlp.models.modelutils.embedding import PositionwiseFeedForward, PositionalEncoding, Embeddings
from dl4nlp.models.modelutils.utils_transfromer import clones


@register_model("transformer")
class Transformer(BaseModel):
    def __init__(self):
        super().__init__()

    @classmethod
    def build_model(self, args, task):
        "Helper: Construct a model from hyperparameters."

        # make sure all arguments are present in older models
        base_architecture(args)
        self.embed_dim = args.encoder_embed_dim

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        context = 5
        N = args.encoder_layers
        d_model = args.encoder_embed_dim
        d_ff = args.encoder_ffn_embed_dim
        h = args.encoder_attention_heads
        dropout = args.dropout
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, context=context)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout, context=context)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, context=context), N),
            nn.Sequential(Embeddings(d_model, len(src_dict)), c(position)),
            nn.Sequential(Embeddings(d_model, len(tgt_dict)), c(position)),
            Generator(d_model, len(tgt_dict), context=context))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src_tokens, src_lengths=None, prev_output_tokens=None, src_mask=None):
        "Take in and process masked src and target sequences."
        # return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        return self.encode(src_tokens, src_mask)

    def encode(self, src, src_mask):
        n_tokens = int(len(src))
        embeds = self.src_embed(src)
        # print(embeds.size())
        # exit()
        # enc = self.encoder(embeds, src_mask)
        # return self.generator(enc)
        enc = self.encoder(embeds.view(n_tokens, 1, -1), src_mask)
        return self.generator(enc.view(n_tokens, -1))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, context=1):
        super(Generator, self).__init__()

        d_model = d_model*context
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=1)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout, context=1):
        super(EncoderLayer, self).__init__()

        size = size*context
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
