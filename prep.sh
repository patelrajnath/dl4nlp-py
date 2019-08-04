#!/usr/bin/env bash
TEXT=/home/raj/PycharmProjects/iconic_fairseq-py/examples/translation/iwslt14.tokenized.de-en
python3.6 preprocess.py --source-lang de --target-lang en \
            --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
                --destdir data-bin/
