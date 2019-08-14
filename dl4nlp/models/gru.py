from torch import nn
import torch.nn.functional as F

from dl4nlp.models import BaseModel, register_model, register_model_architecture


@register_model("gru")
class GRU(BaseModel):
    def __init__(self):
        super().__init__()

    @classmethod
    def build_model(self, args, task):
        # make sure all arguments are present in older models
        base_architecture(args)
        self.embed_dim = args.encoder_embed_dim
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        context = 5
        N = args.encoder_layers
        d_model = args.encoder_embed_dim
        dropout = args.dropout
        bidirectional = args.encoder_bidirectional

        model = GRUModel(num_layers = N,
                         context=context,
                         embedding_dim=self.embed_dim,
                         hidden_dim=d_model,
                         vocab_size= len(src_dict),
                         tagset_size=len(tgt_dict),
                         dropout=dropout,
                         bidirectional=bidirectional)
        return model


class GRUModel(nn.Module):

    def __init__(self, num_layers, context, embedding_dim, hidden_dim, vocab_size, tagset_size, dropout, bidirectional=False):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size + 1, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.dropout_out = dropout
        self.context = context

        self.lstm = nn.GRU(
            input_size=self.context*embedding_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=self.bidirectional)

        # The linear layer that maps from hidden state space to tag space
        if bidirectional:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        n_tokens = int(len(sentence))
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(n_tokens, 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(n_tokens, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


@register_model_architecture('gru', 'gru')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
