import logging
import xgboost
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
import statsmodels.api as sm
import random

from src.functions.printing_and_logging_functions import print_and_log


def raw_features_to_list(final_features):
    raw_features = []
    for feat in final_features[:]:
        if 'binned' in feat:
            prefix, _, _ = str(feat).partition('_binned')
            if '_ratio_' in prefix:
                if '_div_' in prefix:
                    a, b, c = str(prefix).partition('_div_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_add_' in prefix:
                    a, b, c = str(prefix).partition('_add_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_subs_' in prefix:
                    a, b, c = str(prefix).partition('_subs_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_mult_' in prefix:
                    a, b, c = str(prefix).partition('_mult_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
            else:
                raw_features.append(prefix)
        elif '_ratio_' in feat:
            if '_div_' in feat:
                a, b, c = str(feat).partition('_div_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_add_' in feat:
                a, b, c = str(feat).partition('_add_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_subs_' in feat:
                a, b, c = str(feat).partition('_subs_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_mult_' in feat:
                a, b, c = str(feat).partition('_mult_ratio_')
                raw_features.append(a)
                raw_features.append(c)
        else:
            raw_features.append(feat)
    raw_features = list(dict.fromkeys(raw_features))
    return raw_features


def cut_into_bands(X, y, depth):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X, y)
    return clf.predict(X), clf


def get_metrics(y_true, y_pred, y_pred_prob):
    try:
        roc_auc_score_val = round(roc_auc_score(y_true, y_pred_prob), 2)
        y_pred = [round(value) for value in y_pred[:]]
        accuracy_score_val = round(accuracy_score(y_true, y_pred), 2)
        precision_score_val = round(precision_score(y_true, y_pred), 2)
        recall_score_val = round(recall_score(y_true, y_pred), 2)
        f1_score_val = round(f1_score(y_true, y_pred), 2)
        print_and_log(
            f'Metrics: Model found: AS:, {accuracy_score_val}, AUC: {roc_auc_score_val}, '
            f'Precision: {precision_score_val}, Recall: {recall_score_val}, '
            f'F1: {f1_score_val}, df shape: {len(y_true)}', '')
    except Exception as e:
        print_and_log(
            f"Metrics error: {e}. All metrics' values will be set to 0.5. May be the issue is that you have included "
            f"an observation period that has no full performance period and therefore no real cases to be predicted?",
            'RED')
        accuracy_score_val = 0.5
        roc_auc_score_val = 0.5
        precision_score_val = 0.5
        recall_score_val = 0.5
        f1_score_val = 0.5
    return accuracy_score_val, roc_auc_score_val, precision_score_val, recall_score_val, f1_score_val
