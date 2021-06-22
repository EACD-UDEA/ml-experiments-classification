"""
In this module we store functions to measure the performance of our model.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_metric_name_mapping():
    return {_accuracy(): accuracy_score, _f1(): f1_score, _precision(): precision_score, _recall(): recall_score}


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _accuracy(): make_scorer(accuracy_score, greater_is_better=True, **params),
        _f1(): make_scorer(f1_score, greater_is_better=True, **params),
        _precision(): make_scorer(precision_score, greater_is_better=True, **params),
        _recall(): make_scorer(recall_score, greater_is_better=True, **params),
    }
    return mapping[name]


def _accuracy():
    return "accuracy score"

def _f1():
    return "f1 score"

def _precision():
    return "precision score"

def _recall():
    return "recall score"
