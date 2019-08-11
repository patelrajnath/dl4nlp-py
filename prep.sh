#!/usr/bin/env bash
TEXT=examples/pos/icon2016/
TEXT=examples/ner/
python3 preprocess.py --source-lang txt --target-lang tags \
            --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
                --destdir data-bin-ner/
