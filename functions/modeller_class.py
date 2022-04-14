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

class modeller(self):
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.model = None
        self.final_features = None
        self.ac_train, self.auc_train, self.prec_train, self.recall_train, self.f1_train = None
        self.ac_test, self.auc_test, self.prec_test, self.recall_test, self.f1_test = None
        self.ac_t1, self.auc_t1, self.prec_t1, self.recall_t1, self.f1_t1 = None
        self.ac_t2, self.auc_t2, self.prec_t2, self.recall_t2, self.f1_t2 = None
        self.ac_t3, self.auc_t3, self.prec_t3, self.recall_t3, self.f1_t3 = None
        return

    def model_fit(self, train_X, test_X, train_y, test_y):
        self.model.fit(train_X[self.final_features], train_y, eval_set=[(test_X[self.final_features], test_y)],
                        early_stopping_rounds=15)
        return

    def get_metrics(y_true, y_pred, y_pred_prob):
        try:
            roc_auc_score_val = round(roc_auc_score(y_true, y_pred_prob), 2)
            y_pred = [round(value) for value in y_pred[:]]
            accuracy_score_val = round(accuracy_score(y_true, y_pred), 2)
            precision_score_val = round(precision_score(y_true, y_pred), 2)
            recall_score_val = round(recall_score(y_true, y_pred), 2)
            f1_score_val = round(f1_score(y_true, y_pred), 2)
            print(
                f'\t AS:, {accuracy_score_val}, AUC: {roc_auc_score_val}, Precision: {precision_score_val}, Recall: {recall_score_val}, F1: {f1_score_val}, df shape: {len(y_true)}')
            logging.info(
                f'Metrics: Model found: AS:, {accuracy_score_val}, AUC: {roc_auc_score_val}, Precision: {precision_score_val}, Recall: {recall_score_val}, F1: {f1_score_val}, df shape: {len(y_true)}')
        except Exception as e:
            print(
                f"Metrics error: {e}. All metrics' values will be set to 0.5. May be the issue is that you have included "
                f"an observation period that has no full performance period and therefore no real cases to be predicted?")
            accuracy_score_val = 0.5
            roc_auc_score_val = 0.5
            precision_score_val = 0.5
            recall_score_val = 0.5
            f1_score_val = 0.5
        return accuracy_score_val, roc_auc_score_val, precision_score_val, recall_score_val, f1_score_val

    def cut_into_bands(X, y, depth):
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X, y)
        return clf.predict(X), clf

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
        criterion - series, the column that the model is training/predicting
        test_y, test_X - test group used for modelling
        predict only flag - indicates if the function should train or predict
        final features - the features to be used for training/predicting
        cut_points_train - the cut points for probability to generate bands (experimental)
        cut_offs - defined cutoffs in the param file to generate score bands based on the probability
    """
    results = pd.DataFrame()
    if predict_only_flag != 'yes':
        if params['trees_features_to_include']:
            final_features = params['trees_features_to_include']

        xgb_def = xgboost.XGBClassifier()

        if params['under_sampling']:
            print('\n\t *** UNDERSAMPLING MODEL ***')
            xgb_def.fit(df_us[final_features], criterion_us, eval_set=[(test_X_us[final_features], test_y_us)],
                        early_stopping_rounds=15)
        else:
            xgb_def.fit(df[final_features], criterion, eval_set=[(test_X[final_features], test_y)],
                        early_stopping_rounds=15)

        results['columns'] = df[final_features].columns
        results['importances'] = xgb_def.feature_importances_
        results.sort_values(by='importances', ascending=False, inplace=True)
        results = results[results['importances'] > 0]
        results = results[results['importances'] < 0.95]
        results = results[:30]
        final_features = results['columns'].unique().tolist()

        if params['trees_features_to_exclude']:
            trees_features_to_exclude = params['trees_features_to_exclude']
            print(f'\t Removing features specified in the params file: {trees_features_to_exclude}')
            logging.info(f'Removing features specified in the params file: {trees_features_to_exclude}')
            for el in trees_features_to_exclude:
                try:
                    final_features.remove(el)
                    results = results[~results['columns'].str.contains(el)]
                except:
                    pass

        logging.info(f'FINAL FEATURES: {final_features}')




        #TODO: separate

    else:
        xgb_def = model_to_predict
        final_features = xgb_def.get_booster().feature_names

    #TODO: # Move to main.py
    df['xgb_y_pred'] = xgb_def.predict(df[final_features])
    df['xgb_deciles_predict'] = pd.qcut(df['xgb_y_pred'], 10, duplicates='drop', labels=False)
    df['xgb_y_pred_prob'] = xgb_def.predict_proba(df[final_features])[:, 1]
    df['xgb_deciles_pred_prob'] = pd.qcut(df['xgb_y_pred_prob'], 10, duplicates='drop', labels=False)

    ac, auc, prec, recall, f1 = get_metrics(y_pred=df['xgb_y_pred'], y_true=criterion, y_pred_prob=df['xgb_y_pred_prob'])

    if cut_offs["xgb"]:
        df['xgb_bands_predict'] = pd.cut(df['xgb_y_pred'], bins=cut_offs["xgb"], include_lowest=True).astype('str')
        df['xgb_bands_predict_proba'] = pd.cut(df['xgb_y_pred_prob'], bins=cut_offs["xgb"], include_lowest=True).astype(
            'str')
    else:
        df['xgb_bands_predict'], _ = cut_into_bands(X=df[['xgb_y_pred']], y=criterion, depth=3)
        df['xgb_bands_predict_proba'], _ = cut_into_bands(X=df[['xgb_y_pred_prob']], y=criterion, depth=3)

    logging.info('XGB: Model found')
    return df, xgb_def, ac, auc, prec, len(final_features), results, cut_points_train, recall, f1











