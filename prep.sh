#!/usr/bin/env bash
TEXT=examples/pos/icon2016/
python3 preprocess.py --source-lang txt --target-lang tags \
            --trainpref $TEXT/hi-en.train --validpref $TEXT/hi-en.dev --testpref $TEXT/hi-en.test \
                --destdir data-bin/
