import os
from argparse import Namespace

import torch
from torch import nn, optim

from dl4nlp import tasks, options, utils
from dl4nlp.data import iterators
from dl4nlp.models.checkpoint_utils import load_model_state, save_state
from dl4nlp.models.cnn import CNNTagger
from dl4nlp.models.gru import GRUTagger
from dl4nlp.models.lstm import LSTMTagger
from dl4nlp.models.transformer_attn import build_model
from dl4nlp.models.modelutils.utils import contextwin
from dl4nlp.options import add_dataset_args, get_parser


def prepare_sequence(seq, to_ix, ctx=None, use_cuda=False):
    idxs = [to_ix[w] for w in seq]
    if ctx:
        idxs = contextwin(idxs, ctx)

    if use_cuda:
        return torch.tensor(idxs, dtype=torch.long).cuda()

    return torch.tensor(idxs, dtype=torch.long)


def get_data_parser(default_task='translation'):
    parser = get_parser('Trainer', default_task)
    add_dataset_args(parser, train=True)
    return parser


def get_train_iterator(task, args, epoch, combine=True):
    """Return an EpochBatchIterator over the training set for a given epoch."""
    print('| loading train data for epoch {}'.format(epoch))
    task.load_dataset(args.train_subset, epoch=epoch, combine=combine)
    return task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        epoch=epoch,
    )


def get_validation_iterator(task, args, epoch, combine=False):
    """Return an EpochBatchIterator over the training set for a given epoch."""
    print('| loading train data for epoch {}'.format(epoch))
    task.load_dataset(args.valid_subset, epoch=epoch, combine=combine)
    return task.get_batch_iterator(
        dataset=task.dataset(args.valid_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        epoch=epoch,
    )


def prepare_sample(sample, use_cuda=False):
    if sample is None or len(sample) == 0:
        return None
    if use_cuda:
        return torch.tensor(sample, dtype=torch.long).cuda()
    return torch.tensor(sample, dtype=torch.long)

parser = get_data_parser()
args = options.parse_args_and_arch(parser)
task = tasks.setup_task(args)
train_iter = get_train_iterator(task, args, epoch=0, combine=True)

itr = train_iter.next_epoch_itr(
        shuffle=(train_iter.epoch >= args.curriculum),
    )


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 512
NUM_LAYERS = 2
HIDDEN_DIM = 256
CONTEXT=5

# model = LSTMTagger(num_layers=NUM_LAYERS,
#                    context=CONTEXT,
#                    embedding_dim=EMBEDDING_DIM,
#                    hidden_dim=HIDDEN_DIM,
#                    vocab_size=len(task.src_dict),
#                    tagset_size=len(task.tgt_dict),
#                    bidirectional=True)
model = GRUTagger(num_layers=NUM_LAYERS,
                  context=CONTEXT,
                  embedding_dim=EMBEDDING_DIM,
                  hidden_dim=HIDDEN_DIM,
                  vocab_size=len(task.src_dict),
                  tagset_size=len(task.tgt_dict),
                  bidirectional=True)
# model = CNNTagger(num_layers=NUM_LAYERS,
#                   context=CONTEXT,
#                   embedding_dim=EMBEDDING_DIM,
#                   hidden_dim=HIDDEN_DIM,
#                   vocab_size=len(task.src_dict),
#                   tagset_size=len(task.tgt_dict))
# model = build_model(src_vocab=len(task.src_dict),
#                     tgt_vocab=len(task.tgt_dict),
#                     context=CONTEXT,
#                     N=NUM_LAYERS)

# model = GRUTagger(NUM_LAYERS, CONTEXT, EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
# model = CNNTagger(NUM_LAYERS, CONTEXT, EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
# model = build_model(len(word_to_ix), len(tag_to_ix), context=CONTEXT, N=1)

loss_function = nn.NLLLoss()
# loss_function = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
# optimizer = optim.Adadelta(model.parameters())

use_cuda = torch.cuda.is_available()
print(use_cuda)
# use_cuda=False

if use_cuda:
    model.cuda()
    # optimizer.cuda()
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
training = False
modeldir="transformer-models"
modeldir="lstm-models"
modeldir="gru-models"
if not os.path.exists(modeldir):
    os.mkdir(modeldir)

checkpoint_last = 'checkpoint_last.pt'
if training:
    start_epoch = load_model_state(os.path.join(modeldir, checkpoint_last), model)
    for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
        itr = train_iter.next_epoch_itr(shuffle=(train_iter.epoch >= args.curriculum))
        itr = iterators.GroupedIterator(itr, 1)
        model.zero_grad()
        for i, samples in enumerate(itr):
            for j, sample in enumerate(samples):
                net_input = prepare_sample(contextwin(sample['net_input']['src_tokens'].tolist()[0], CONTEXT), use_cuda)
                print(net_input)
                # tag_scores = model(**sample['net_input'])
                tag_scores = model(net_input)
                tag_scores = tag_scores.view(-1, tag_scores.size(-1))
                target = sample['target'].view(-1).cuda()
                print(tag_scores.size())
                print(target.size())
                loss = loss_function(tag_scores, target)
                print("The Loss", loss)
                loss.backward()
                optimizer.step()
        checkpoint = "checkpoint" + str(epoch) + ".pt"
        save_state(os.path.join(modeldir, checkpoint), model, loss_function, optimizer, epoch)
        save_state(os.path.join(modeldir, checkpoint_last), model, loss_function, optimizer, epoch)
        # print(out)

# for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model.zero_grad()
#
#         # Step 2. Get our inputs ready for the network, that is, turn them into
#         # Tensors of word indices.
#         sentence_in = prepare_sequence(sentence, word_to_ix, CONTEXT, use_cuda=use_cuda)
#         # sentence_in = prepare_sequence(sentence, word_to_ix, use_cuda=use_cuda)
#         targets = prepare_sequence(tags, tag_to_ix, use_cuda=use_cuda)
#
#         # Step 3. Run our forward pass.
#         tag_scores = model(sentence_in)
#         print(tag_scores)
#
#         # Step 4. Compute the loss, gradients, and update the parameters by
#         #  calling optimizer.step()
#         loss = loss_function(tag_scores, targets)
#         print(loss)
#         loss.backward()
#         optimizer.step()

# See what the scores are after training
itr = get_validation_iterator(task, args, epoch=0, combine=True).next_epoch_itr(shuffle=False)
itr = iterators.GroupedIterator(itr, 1)

with torch.no_grad():
    # inputs = prepare_sequence(training_data[0][0], word_to_ix, CONTEXT, use_cuda=use_cuda)
    # inputs = prepare_sequence(training_data[0][0], word_to_ix, use_cuda=use_cuda)
    # tag_scores = model(inputs)
    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    start_epoch = load_model_state(os.path.join(modeldir, checkpoint_last), model)
    for i, samples in enumerate(itr):
        print(i)
        count = 0
        correct_score = 0
        for j, sample in enumerate(samples):
            net_input = prepare_sample(contextwin(sample['net_input']['src_tokens'].tolist()[0], CONTEXT), use_cuda)
            # print(**sample['net_input'])
            tag_scores = model(net_input)
            tag_scores = tag_scores.view(-1, tag_scores.size(-1))
            target = sample['target'].view(-1).cuda()
            _, predicted = torch.max(tag_scores, dim=1)
            correct = torch.nonzero(predicted == target)
            print(target.size(0))
            correct_score += (correct.size(0)*1.0/target.size(0))*100
            count +=1

    print("Accuracy", correct_score/count)