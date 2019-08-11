import sys

fin = open(sys.argv[1])
tag = list()
text = list()
print(sys.argv[1])
out = sys.argv[1]
with open('valid.txt', 'w') as fout_text, \
        open('valid.tags', 'w') as fout_tag:
    for line in fin:
        line = line.strip()
        if not line:
            print(' '.join(text))
            fout_text.write(' '.join(text) + '\n')
            fout_tag.write(' '.join(tag) + '\n')
            tag = list()
            text = list()
            continue
        word, pos = line.split('\t')
        tag.append(pos)
        text.append(word)