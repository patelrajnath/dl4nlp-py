# dl4nlp-py (Deep Learning for Natural Language Processing)
This repository contains:

(1) RNN, CNN, and Transformer based system for word level quality estimation.

(2) RNN, CNN, and Transformer based Part-of-Speech tagger for code-mixed social media text.

The RNN models include simple Recurrent Neural Network, Long-Short Term Memory (LSTM), DeepLSTM,
and Gated Recurrent Units (GRU) aka Gated Hidden Units (GHU).
The system is flexible to be used for any word level NLP tagging task like Named Entity Recognition etc.

## Pre-requisites
- python (3.6+)
- pytorch (1.0+; https://pytorch.org/get-started/locally/)
- numpy
- python-sklearn

## Quick Start

#### POS tagging:

## Publications:

If you use this project, please cite the following papers:

> @InProceedings{patel-m:2016:WMT,
>  author    = {Patel, Raj Nath  and  M, Sasikumar},
>  title     = {Translation Quality Estimation using Recurrent Neural Network},
>  booktitle = {Proceedings of the First Conference on Machine Translation},
>  month     = {August},
>  year      = {2016},
>  address   = {Berlin, Germany},
>  publisher = {Association for Computational Linguistics},
>  pages     = {819--824},
>  url       = {http://www.statmt.org/wmt16/pdf/W16-2389.pdf } }


> @article{patel2016recurrent,
>  title={Recurrent Neural Network based Part-of-Speech Tagger for Code-Mixed Social Media Text},
>  author={Patel, Raj Nath and Pimpale, Prakash B and Sasikumat, M},
>  journal={arXiv preprint arXiv:1611.04989},
>  year={2016}
>  url = {https://arxiv.org/pdf/1611.04989.pdf } }


## Author

Raj Nath Patel (patelrajnath@gmail.com)

Linkedin: https://www.linkedin.com/in/raj-nath-patel-2262b024/

## Version

0.1

## LICENSE

Copyright Raj Nath Patel 2019 - present

rnn4nlp is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

You should have received a copy of the GNU General Public License along with Indic NLP Library. If not, see http://www.gnu.org/licenses/.