from sklearn.metrics import *

def roc_auc_macro(y_true, y_score):
    return roc_auc_score(y_true, y_score, average="macro")


def roc_auc_micro(y_true, y_score):
    return roc_auc_score(y_true, y_score, average="micro")

# TODO: add the roc auc of classes too

def precision_micro(y_true, y_pred, labels=None, sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate prec for neg class
    '''
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=1,
                                                 average='micro',
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return p

def precision_macro(y_true, y_pred, labels=None, sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate prec for neg class
    '''
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=1,
                                                 average='macro',
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return p

def precision_neg(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate prec for neg class
    '''
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return p
def recall_micro(y_true, y_pred, labels=None, sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate recall for neg class
    '''
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=1,
                                                 average='micro',
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return r

def recall_macro(y_true, y_pred, labels=None, sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate recall for neg class
    '''
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=1,
                                                 average='macro',
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return r

def recall_neg(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate recall for neg class
    '''
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return r

def f1_micro(y_true, y_pred, labels=None, sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate f1 for neg class
    '''
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=1,
                                                 average='micro',
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return f

def f1_macro(y_true, y_pred, labels=None, sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate f1 for neg class
    '''
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=1,
                                                 average='macro',
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return f

def f1_neg(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate f1 for neg class
    '''
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return f

class Scoring(object):
    '''
     Scoring scheme
    '''

    metrics = {'auc_score_macro': make_scorer(roc_auc_macro),
               'auc_score_micro': make_scorer(roc_auc_micro),
               'accuracy': 'accuracy',
               'p_pos': 'precision',
               'r_pos': 'recall',
               'f1_pos': 'f1',
               'f1_neg': make_scorer(f1_neg),
               'p_neg': make_scorer(precision_neg),
               'r_neg': make_scorer(recall_neg),
               'precision_micro': 'precision_micro',
               'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro',
               'recall_micro': 'recall_micro', 'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro'}

    functions = {'auc_score_macro': roc_auc_macro,
               'auc_score_micro': roc_auc_micro,
               'accuracy': accuracy_score,
               'p_pos': precision_score,
               'r_pos': recall_score,
               'f1_pos': f1_score,
               'f1_neg': f1_neg,
               'p_neg': precision_neg,
               'r_neg': recall_neg,
               'precision_micro': precision_micro,
               'precision_macro': precision_macro, 'recall_macro': recall_macro,
               'recall_micro': recall_micro, 'f1_macro': f1_macro, 'f1_micro': f1_micro}
