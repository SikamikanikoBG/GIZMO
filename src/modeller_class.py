import logging
import xgboost
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
import statsmodels.api as sm
import random

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)









def xgb(df, criterion, test_X, test_y, df_us, criterion_us, test_X_us, test_y_us, predict_only_flag, model_to_predict,
        final_features, cut_points_train, cut_offs, params):
    """
    Generates the XGBoost model or applies it on a dataframe
    Returns df, xgb_def, ac, auc, prec, len(final_features), results, cut_points_train
        df - the df parameter, enriched with additional columns
        xgb_def - XGBoost model object
        ac - Accuracy score
        AUC - Area under the curve
        prec - Precision score
        len final features - the nb of the features in the model
        results - dataframe with feature names and importance
        cut points train - obsolete

    Parameters:
        df, criterion, test_X, test_y, predict_only_flag, model_to_predict, final_features, cut_points_train, cut_offs
        df - the dataframe on which to train or predict the model
        criterion - series, the column that the model is training_flows/predicting
        test_y, test_X - test group used for modelling
        predict only flag - indicates if the function should train or predict
        final features - the features to be used for training_flows/predicting
        cut_points_train - the cut points for probability to generate bands (experimental)
        cut_offs - defined cutoffs in the param file to generate score bands based on the probability
    """

    return df, xgb_def, ac, auc, prec, len(final_features), results, cut_points_train, recall, f1



    return ac, auc, f1, prec, recall











