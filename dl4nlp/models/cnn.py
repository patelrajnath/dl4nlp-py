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

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1,
                              out_channels=16,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

        # Fully Connected 1
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

        # The linear layer that maps from hidden state space to tag space
        if bidirectional:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        n_tokens = int(len(sentence))
        embeds = self.word_embeddings(sentence)
        # Convolution 1
        out = self.cnn1(embeds.view(n_tokens, 1, -1))
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        out = out.view(out.size(0), -1)

        # Dropout
        out = self.dropout(out)

        # Fully connected 1
        out = self.fc1(out)
        return out

        # tag_space = self.hidden2tag(lstm_out.view(n_tokens, -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores