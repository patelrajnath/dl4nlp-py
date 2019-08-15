#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 train.py data-bin --max-sentences 1 --arch transformer \
	--optimizer adam \
	--adam-betas '(0.9,0.98)' \
	--clip-norm 0.0 \
	--lr 0.0001 \
	--min-lr 1e-09 \
	--lr-scheduler inverse_sqrt
