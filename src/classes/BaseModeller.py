import sys

import pandas as pd
import xgboost
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from src import print_and_log, cut_into_bands, get_metrics


class BaseModeller:
    def __init__(self, model_name, params, final_features, cut_offs):
        self.model_name = model_name
        self.params = params
        self.model = None
        self.final_features = final_features
        self.ac_train, self.auc_train, self.prec_train, self.recall_train, self.f1_train = None, None, None, None, None
        self.ac_test, self.auc_test, self.prec_test, self.recall_test, self.f1_test = None, None, None, None, None
        self.ac_t1, self.auc_t1, self.prec_t1, self.recall_t1, self.f1_t1 = None, None, None, None, None
        self.ac_t2, self.auc_t2, self.prec_t2, self.recall_t2, self.f1_t2 = None, None, None, None, None
        self.ac_t3, self.auc_t3, self.prec_t3, self.recall_t3, self.f1_t3 = None, None, None, None, None
        self.trees_features_to_exclude = self.params['trees_features_to_exclude']
        self.trees_features_to_include = self.params['trees_features_to_include']
        self.cut_offs = self.params["cut_offs"][model_name]
        self.metrics = pd.DataFrame()

    def model_fit(self, train_X, train_y, test_X, test_y):
        if self.model_name == 'xgb':
            self.model.fit(train_X[self.final_features], train_y, eval_set=[(test_X[self.final_features], test_y)],
                           early_stopping_rounds=15)
        else:
            self.model.fit(train_X[self.final_features], train_y)

    def load_model(self):
        if self.model_name == 'xgb':
            self.model = xgboost.XGBClassifier()
        elif self.model_name == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, n_jobs=3)
        elif self.model_name == 'dt':
            self.model = tree.DecisionTreeClassifier(max_depth=4)
        else:
            print_and_log('No model provided (model_name)', 'RED')
            sys.exit()

    def generate_predictions_and_metrics(self, y_true, df):

        df[f'{self.model_name}_y_pred'] = self.model.predict(df[self.final_features])
        df[f'{self.model_name}_deciles_predict'] = pd.qcut(df[f'{self.model_name}_y_pred'], 10, duplicates='drop',
                                                           labels=False)
        df[f'{self.model_name}_y_pred_prob'] = self.model.predict_proba(df[self.final_features])[:, 1]
        df[f'{self.model_name}_deciles_pred_prob'] = pd.qcut(df[f'{self.model_name}_y_pred_prob'], 10,
                                                             duplicates='drop', labels=False)

        if self.cut_offs:
            df[f'{self.model_name}_bands_predict'] = pd.cut(df[f'{self.model_name}_y_pred'], bins=self.cut_offs,
                                                            include_lowest=True).astype(
                'str')
            df[f'{self.model_name}_bands_predict_proba'] = pd.cut(df[f'{self.model_name}_y_pred_prob'],
                                                                  bins=self.cut_offs,
                                                                  include_lowest=True).astype('str')
        else:
            df[f'{self.model_name}_bands_predict'], _ = cut_into_bands(X=df[[f'{self.model_name}_y_pred']], y=y_true,
                                                                       depth=3)
            df[f'{self.model_name}_bands_predict_proba'], _ = cut_into_bands(X=df[[f'{self.model_name}_y_pred_prob']],
                                                                             y=y_true, depth=3)

        ac, auc, prec, recall, f1 = get_metrics(y_pred=df[f'{self.model_name}_y_pred'], y_true=y_true,
                                                y_pred_prob=df[f'{self.model_name}_y_pred_prob'])

        metrics_df = pd.DataFrame()
        metrics_df['AccuracyScore'] = [ac]
        metrics_df['AUC'] = [auc]
        metrics_df['PrecisionScore'] = [prec]
        metrics_df['Recall'] = [recall]
        metrics_df['F1'] = [f1]
        return metrics_df