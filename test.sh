#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 train.py data-bin --max-sentences 1 --valid-subset test
