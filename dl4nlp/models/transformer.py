#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dl4nlp-py
Create bt raj at 2:57 PM,  8/23/19
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from dl4nlp.models import register_model, BaseModel, register_model_architecture


@register_model("transformer")
class Transformer(BaseModel):
    def __init__(self, k, emb_dim, heads, depth, seq_length, num_tokens, num_classes, context=1):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = nn.Embedding(seq_length, emb_dim)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

    @classmethod
    def build_model(self, args, task):
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
        self.max_source_positions = args.max_source_positions
        print("In model build", self.max_source_positions)

        model = self(k=d_model,
                     emb_dim= self.embed_dim,
                     heads=h,
                     depth=N,
                     seq_length=self.max_source_positions,
                     num_tokens=len(src_dict),
                     num_classes=len(tgt_dict),
                     context=context)
        return model

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, e = tokens.size()

        # generate position embeddings
        # positions = torch.arange(t)
        # positions = self.pos_emb(positions)[None, :, :].expand(b, t, e)

        # x = tokens + positions
        x = tokens
        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class
        # probabilities
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super(SelfAttention, self).__init__()

        self.heads = heads

        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        # This unifies the outputs of the different heads into
        # a single k-vector
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights

        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
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
