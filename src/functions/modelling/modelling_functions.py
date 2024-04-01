import traceback
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn import tree

from src.functions.printing_and_logging import print_and_log


def cut_into_bands(X, y, depth):
    """
    Fit a decision tree classifier to the data and predict the target variable.

    Parameters:
    - X: array-like, input features
    - y: array-like, target variable
    - depth: int, maximum depth of the decision tree

    Returns:
    - y_pred: array, predicted target variable
    - clf: DecisionTreeClassifier, trained decision tree classifier
    """

    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X, y)
    return clf.predict(X), clf


def get_multiclass_metrics(y_true, y_pred, y_pred_prob):
    """
    Calculate multiclass classification metrics based on true labels, predicted labels, and predicted probabilities.

    Steps:
    1. Calculate ROC AUC score with macro averaging and 'ovr' strategy.
    2. Round predicted labels to integers.
    3. Calculate accuracy, precision, recall, and F1 score with macro averaging.
    4. Calculate the criterion rate, predicted criterion rate, and total volumes.
    5. Handle exceptions by logging errors and setting metrics to 0.

    Parameters:
    - y_true: array-like, true target labels
    - y_pred: array-like, predicted target labels
    - y_pred_prob: array-like, predicted probabilities for each class

    Returns:
    - accuracy_score_val: float, accuracy score
    - roc_auc_score_val: float, ROC AUC score
    - precision_score_val: float, precision score
    - recall_score_val: float, recall score
    - f1_score_val: float, F1 score
    - criterion_rate: float, true positive rate
    - criterion_rate_pred: float, predicted positive rate
    - volumes: int, total number of samples
    """
    try:
        roc_auc_score_val = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr').round(2)
        y_pred = [round(value) for value in y_pred[:]]
        accuracy_score_val = accuracy_score(y_true, y_pred).round(2)
        precision_score_val = precision_score(y_true, y_pred, average='macro').round(2)
        recall_score_val = recall_score(y_true, y_pred, average='macro').round(2)
        f1_score_val = f1_score(y_true, y_pred, average='macro').round(2)
        criterion_rate = round(sum(y_true) / len(y_true), 2)
        criterion_rate_pred = round(sum(y_pred) / len(y_pred), 2)
        volumes = len(y_true)
    except Exception as e:
        print_and_log(
            f"Multiclass Metrics error: {e}\n. \nAll metrics' values will be set to 0. May be the issue is that you have included "
            f"an observation period that has no full performance period and therefore no real cases to be predicted?",
            'RED')
        traceback.print_exc()
        accuracy_score_val = 0
        roc_auc_score_val = 0
        precision_score_val = 0
        recall_score_val = 0
        f1_score_val = 0
        criterion_rate = 0
        criterion_rate_pred = 0
        volumes = 0

    return accuracy_score_val, roc_auc_score_val, precision_score_val, recall_score_val, f1_score_val, criterion_rate, \
        criterion_rate_pred, volumes


def get_metrics(y_true, y_pred, y_pred_prob):
    """
    Calculate classification metrics based on true labels, predicted labels, and predicted probabilities.

    Steps:
    1. Calculate ROC AUC score.
    2. Round predicted labels to integers.
    3. Calculate accuracy, precision, recall, and F1 score.
    4. Calculate the criterion rate, predicted criterion rate, and total volumes.
    5. Handle exceptions by logging errors and setting metrics to 0.

    Parameters:
    - y_true: array-like, true target labels
    - y_pred: array-like, predicted target labels
    - y_pred_prob: array-like, predicted probabilities for each class

    Returns:
    - accuracy_score_val: float, accuracy score
    - roc_auc_score_val: float, ROC AUC score
    - precision_score_val: float, precision score
    - recall_score_val: float, recall score
    - f1_score_val: float, F1 score
    - criterion_rate: float, true positive rate
    - criterion_rate_pred: float, predicted positive rate
    - volumes: int, total number of samples
    """

    try:
        roc_auc_score_val = round(roc_auc_score(y_true, y_pred_prob), 2)
        y_pred = [round(value) for value in y_pred[:]]
        accuracy_score_val = round(accuracy_score(y_true, y_pred), 2)
        precision_score_val = round(precision_score(y_true, y_pred), 2)
        recall_score_val = round(recall_score(y_true, y_pred), 2)
        f1_score_val = round(f1_score(y_true, y_pred), 2)
        criterion_rate = round(sum(y_true) / len(y_true), 2)
        criterion_rate_pred = round(sum(y_pred) / len(y_pred), 2)
        volumes = len(y_true)
    except Exception as e:
        print_and_log(
            f"Metrics error: {e}. All metrics' values will be set to 0. May be the issue is that you have included "
            f"an observation period that has no full performance period and therefore no real cases to be predicted?",
            'RED')
        accuracy_score_val = 0
        roc_auc_score_val = 0
        precision_score_val = 0
        recall_score_val = 0
        f1_score_val = 0
        criterion_rate = 0
        criterion_rate_pred = 0
        volumes = 0

    return accuracy_score_val, roc_auc_score_val, precision_score_val, recall_score_val, f1_score_val, criterion_rate, \
        criterion_rate_pred, volumes
