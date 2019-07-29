from torch import nn
import torch.nn.functional as F


class GRUTagger(nn.Module):

    def __init__(self, num_layers, context, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=False):
        super(GRUTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size + 1, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.dropout_out = 0.01
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