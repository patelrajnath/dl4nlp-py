from sklearn.metrics import f1_score


def weighted_fmeasure(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    return f1_score(y_true, y_pred, average='weighted', pos_label=None)


def get_f1_score(hyp, ref):
    """
    :param hyp:
    :param ref:
    :return:
    """
    return f1_score(ref, hyp, average=None), weighted_fmeasure(ref, hyp)