import os

import numpy
import torch
from torch import nn

from dl4nlp import tasks, options, utils, optim
from dl4nlp.data import iterators
from dl4nlp.eval.f1_measure import get_f1_score
from dl4nlp.models.checkpoint_utils import load_model_state, save_state

# This is required for registering the models
from dl4nlp.models.cnn import CNNTagger
from dl4nlp.models.lstm import LSTMTagger
from dl4nlp.models.transformer import Transformer2
from dl4nlp.models.transformer_att import Transformer

from dl4nlp.models.modelutils.utils import contextwin
from dl4nlp.options import add_dataset_args, get_parser, add_model_args, add_optimization_args, add_checkpoint_args, \
    add_distributed_training_args
from dl4nlp.logger import LogManager

logger = LogManager().logger


def prepare_sequence(seq, to_ix, ctx=None, use_cuda=False):
    idxs = [to_ix[w] for w in seq]
    if ctx:
        idxs = contextwin(idxs, ctx, pad_id=0)

    if use_cuda:
        return torch.tensor(idxs, dtype=torch.long).cuda()

    return torch.tensor(idxs, dtype=torch.long)


def get_data_parser(default_task='translation'):
    parser = get_parser('Trainer', default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
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

CONTEXT=5
model = task.build_model(args)

# loss_function = nn.NLLLoss()
# CNN Training
loss_function = nn.CrossEntropyLoss()
optimizer = optim.build_optimizer(args, model.parameters())

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
# modeldir="transformer-models"
# modeldir="lstm-models"
modeldir="gru-models"
# modeldir="cnn-models"
if not os.path.exists(modeldir):
    os.mkdir(modeldir)
checkpoint_last = 'checkpoint_last.pt'
checkpoint_best = "checkpoint_best.pt"


# See what the scores are after training
def get_accuracy_scores(eval=False):
    data_iterator = get_validation_iterator(task, args, epoch=0, combine=True).next_epoch_itr(shuffle=False)
    data_iterator = iterators.GroupedIterator(data_iterator, 1)
    if eval:
        start_epoch = load_model_state(os.path.join(modeldir, checkpoint_best), model)
    with torch.no_grad():
        hypothesis = list()
        reference = list()
        for i, samples in enumerate(data_iterator):
            for j, sample in enumerate(samples):
                net_input = prepare_sample(contextwin(sample['net_input']['src_tokens'].tolist()[0], CONTEXT,
                                                      pad_id=task.tgt_dict.pad()),
                                           use_cuda)
                # print(**sample['net_input'])
                tag_scores = model(net_input)
                tag_scores = tag_scores.view(-1, tag_scores.size(-1))
                if use_cuda:
                    target = sample['target'].view(-1).cuda()
                else:
                    target = sample['target'].view(-1)
                _, predicted = torch.max(tag_scores, dim=1)
                hypothesis.extend(predicted.tolist())
                reference.extend(target.tolist())
        return get_f1_score(reference, hypothesis)


training = True
# train with early stopping on validation set
best_f1 = -numpy.inf
training_options = dict()
if training:
    start_epoch = load_model_state(os.path.join(modeldir, checkpoint_last), model)
    for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
        training_options["ce"] = epoch
        itr = train_iter.next_epoch_itr(shuffle=(train_iter.epoch >= args.curriculum))
        itr = iterators.GroupedIterator(itr, 1)
        model.zero_grad()

        epoch_loss = 0
        for i, samples in enumerate(itr):
            for j, sample in enumerate(samples):
                net_input = prepare_sample(
                    contextwin(l=sample['net_input']['src_tokens'].tolist()[0],
                               win=CONTEXT,
                               pad_id=task.tgt_dict.pad()
                               ),
                    use_cuda=use_cuda
                )
                # tag_scores = model(**sample['net_input'])
                tag_scores = model(net_input)
                tag_scores = tag_scores.view(-1, tag_scores.size(-1))
                if use_cuda:
                    target = sample['target'].view(-1).cuda()
                else:
                    target = sample['target'].view(-1)
                loss = loss_function(tag_scores, target)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
        print("epoch {0} loss={1}".format(epoch, epoch_loss))
        validation = get_accuracy_scores()
        checkpoint = "checkpoint" + str(epoch) + ".pt"
        save_state(os.path.join(modeldir, checkpoint), model, loss_function, optimizer, epoch)
        save_state(os.path.join(modeldir, checkpoint_last), model, loss_function, optimizer, epoch)
        if validation[1] > best_f1:
            logger.info('NEW BEST: epoch' + str(epoch) + ', valid F1=' +  str(validation[1]))
            save_state(os.path.join(modeldir, checkpoint_best), model, loss_function, optimizer, epoch)
            training_options["be"] = epoch
            best_f1 = validation[1]
            # Break if no improvement in 10 epochs
        if abs(training_options['be'] - training_options['ce']) >= 10:
            break
    logger.info('BEST RESULT: epoch' + str(training_options['be']) + ', valid F1=' + str(best_f1) + ', final checkpoint-' + checkpoint_best)

print(get_accuracy_scores(eval=True))
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
