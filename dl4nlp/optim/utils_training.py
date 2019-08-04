#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
 Copyright (c) 2018 Raj Nath Patel
 Licensed under the GNU Public License.
 Author: Raj Nath Patel
 Email: patelrajnath (at) gmail (dot) com
 Created: 25/May/2018 03:48
 """

import time
import torch
import os
import logging
import traceback


from torch.autograd import Variable
from torch.serialization import default_restore_location

from dl4nlp.models.modelutils.utils_transfromer import subsequent_mask


class Batch:
    "Object for holding a batch of modelutils with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


def run_epoch(data_iter, model, loss_compute, epoch=0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start
            print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                    (epoch, i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, model, criterion, optimizer,
               num_updates, optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    print("Saving checkpoint at-", filename)
    state_dict = {
        'model': model.state_dict(),
        'num_updates': num_updates,
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
            }
        ],
        'extra_state': extra_state,
    }
    torch_persistent_save(state_dict, filename)


def load_model_state(filename, model, cuda_device=None):
    if not os.path.exists(filename):
        print("Starting training from scratch.")
        return 0

    print("Loading model from checkpoints", filename)
    if cuda_device is None:
        state = torch.load(filename)
    else:
        state = torch.load(
            filename,
            map_location=lambda s, l: default_restore_location(s, 'cuda:{}'.format(cuda_device))
        )

    # load model parameters
    try:
        model.load_state_dict(state['model'])
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return state['num_updates']
