#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 train.py data-bin --max-sentences 1 --arch transformer