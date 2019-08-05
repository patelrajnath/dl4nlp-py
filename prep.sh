#!/usr/bin/env bash
TEXT=/home/raj/PycharmProjects/dl4nlp-py/examples/pos/icon2016/
python3.6 preprocess.py --source-lang txt --target-lang tags \
            --trainpref $TEXT/hi-en.train --validpref $TEXT/hi-en.dev --testpref $TEXT/hi-en.test \
                --destdir data-bin/
