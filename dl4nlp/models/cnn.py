from torch import nn
import torch.nn.functional as F

from dl4nlp.models import register_model, BaseModel, register_model_architecture


@register_model("cnn")
class CNN(BaseModel):
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
        model = CNNTagger(num_layers = N,
                         context=context,
                         embedding_dim=self.embed_dim,
                         hidden_dim=d_model,
                         vocab_size= len(src_dict),
                         tagset_size=len(tgt_dict),
                         dropout=dropout,
                         bidirectional=bidirectional)
        return model


class CNNTagger(nn.Module):

    def __init__(self, num_layers, context, embedding_dim, hidden_dim, vocab_size, tagset_size, dropout, bidirectional=False):
        super(CNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.dropout_out = dropout
        self.context = context
        self.in_channel=self.context*embedding_dim

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=self.in_channel,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_out)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        n_tokens = int(len(sentence))
        embeds = self.word_embeddings(sentence)
        # Convolution 1
        out = self.cnn1(embeds.view(n_tokens, self.in_channel, 1, -1))
        # out = self.cnn1(embeds)
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
        tag_scores = self.hidden2tag(out)
        return tag_scores

        # tag_space = self.hidden2tag(lstm_out.view(n_tokens, -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores


@register_model_architecture('cnn', 'cnn')
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