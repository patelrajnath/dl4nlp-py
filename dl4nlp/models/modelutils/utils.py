

def contextwin(l, win, pad_id):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = int(win/2) * [pad_id] + l + int(win/2) * [pad_id]
    out = [lpadded[i:i+win] for i in range(len(l))]

    assert len(out) == len(l)
    return out
