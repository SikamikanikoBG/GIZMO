import sys
import definitions

import pandas as pd
import xgboost
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from src import print_and_log, cut_into_bands, get_metrics, get_multiclass_metrics


class BaseModeller:
    """
    Initialize the BaseModeller object with model information, parameters, features, and cut-offs.

    Args:
        model_name (str): Name of the model.
        params (dict): Dictionary of parameters.
        final_features (list): List of final features.
        cut_offs: Cut-off values.

    Attributes:
        model_name (str): Name of the model.
        params (dict): Dictionary of parameters.
        model: Placeholder for the model.
        final_features (list): List of final features.
        ac_train, auc_train, prec_train, recall_train, f1_train: Evaluation metrics for training data.
        ac_test, auc_test, prec_test, recall_test, f1_test: Evaluation metrics for test data.
        ac_t1, auc_t1, prec_t1, recall_t1, f1_t1: Evaluation metrics for t1 data.
        ac_t2, auc_t2, prec_t2, recall_t2, f1_t2: Evaluation metrics for t2 data.
        ac_t3, auc_t3, prec_t3, recall_t3, f1_t3: Evaluation metrics for t3 data.
        trees_features_to_exclude: Features to exclude for tree models.
        trees_features_to_include: Features to include for tree models.
        lr_features: Features for logistic regression.
        lr_features_to_include: Features to include for logistic regression.
        cut_offs: Cut-off values.
        metrics: DataFrame to store metrics.
        lr_logit_roc_auc: Placeholder for logistic regression ROC AUC score.
        lr_table: Placeholder for logistic regression summary table.
    """
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
        self.lr_features = self.params['lr_features']
        self.lr_features_to_include = self.params['lr_features_to_include']
        self.cut_offs = self.params["cut_offs"][model_name]
        self.metrics = pd.DataFrame()
        self.lr_logit_roc_auc = None
        self.lr_table = None

    def model_fit(self, train_X, train_y, test_X, test_y):
        """
        Fit the model using the training data and evaluate on the test data.

        Args:
            train_X: Training features.
            train_y: Training target.
            test_X: Test features.
            test_y: Test target.
        """
        if self.model_name == 'xgb':
            eval_metric = ['auc', 'error', 'logloss']
            if (train_y.nunique() > 2):
                eval_metric = ['auc', 'merror', 'mlogloss']
            
            self.model.fit(train_X[self.final_features], train_y,
                           eval_set=[(train_X[self.final_features], train_y), (test_X[self.final_features], test_y)],
                           early_stopping_rounds=definitions.early_stopping_rounds, verbose=False, eval_metric=eval_metric)
        elif self.model_name == 'rf':
            self.model.fit(train_X[self.final_features], train_y)
        elif self.model_name == 'dt':
            self.model.fit(train_X[self.final_features], train_y)
        elif self.model_name == 'lr':
            pass
            """train_X = sm.add_constant(train_X)  # add constant
            self.model = sm.Logit(train_y, train_X)  # add model
            self.model = self.model.fit(disp=False)  # fit model
            self.lr_logit_roc_auc = roc_auc_score(train_y, self.model.predict(train_X))
            table = self.model.summary()"""
        else:
            print_and_log(f"[ Modelling ] ERROR: Model {self.model_name} not recognized.", "RED")
            quit()

    def load_model(self):
        """
        Load the model based on the specified model name.
        """
        if self.model_name == 'xgb':
            self.model = xgboost.XGBClassifier(n_estimators=definitions.n_estimators, colsample_bytree=.1, subsample=.5, learning_rate=definitions.learning_rate)
        elif self.model_name == 'rf':
            self.model = RandomForestClassifier(n_estimators=300,
                                                random_state=42,
                                                max_depth=5,
                                                n_jobs=3,
                                                bootstrap=False)
        elif self.model_name == 'dt':
            self.model = tree.DecisionTreeClassifier(max_depth=4)
        elif self.model_name == 'lr':
            pass
        else:
            print_and_log('ERROR: No model provided (model_name)', 'RED')
            sys.exit()

    def generate_multiclass_predictions_and_metrics(self, y_true, df, classes, desired_cutoff=0.5):
        """
       Generate multiclass predictions and metrics based on the input data.

       Args:
           y_true: True target values.
           df: Input DataFrame.
           classes: Classes information.
           desired_cutoff: Desired cutoff value.

       Returns:
           metrics: DataFrame with evaluation metrics.
           df: Updated DataFrame with predictions and probabilities.
       """
        metrics = pd.DataFrame()

        y_pred = self.model.predict(df[self.final_features])
        deciles_predict = pd.qcut(y_pred, 10, duplicates='drop', labels=False)
        y_pred_prob = self.model.predict_proba(df[self.model.get_booster().feature_names])
        
        pred_classes = [f"proba_for_criterion_class_{c}" for c in classes['class_label']]
        y_pred_prob = pd.DataFrame(y_pred_prob, columns=pred_classes, index=y_true.index)
        
        df = pd.concat([df, y_pred_prob], axis=1)

        dec_classes = [f"criterion_decil_class_{c}" for c in classes['class_label']]
        deciles_pred_prob = pd.DataFrame([], columns=dec_classes)
        for cl in classes['class_label']: 
            deciles_pred_prob[f"criterion_decil_class_{cl}"]  = pd.qcut(y_pred_prob[f'proba_for_criterion_class_{cl}'], 10, duplicates='drop', labels=False)
        
        df = pd.concat([df, deciles_pred_prob], axis=1)

        # TODO find out why storing these corrupts the feather file
        bands_predict = []
        if self.cut_offs:
            bands_predict = pd.cut(y_pred, bins=self.cut_offs,include_lowest=True)
        else:
            bands_predict  = cut_into_bands(y_pred, y_true, depth=3)

#        df_temp_tocalc_proba = df[df[f'{self.model_name}_y_pred_prob'] >= desired_cutoff].copy()
#        cr_p_des_cutt = round(df_temp_tocalc_proba[self.params['criterion_column']].sum() / df_temp_tocalc_proba[
#            self.params['criterion_column']].count(), 2)
#        cr_p_des_vol = round( 
#           df_temp_tocalc_proba[f'{self.model_name}_y_pred'].sum() / df[self.params['criterion_column']].count(), 2)

        ac, auc, prec, recall, f1, cr, cr_p, vol = get_multiclass_metrics(y_true, y_pred,  y_pred_prob)
        metrics['AccuracyScore'] = [ac]
        metrics['AUC'] = [auc]
        metrics['PrecisionScore'] = [prec]
        metrics['Recall'] = [recall]
        metrics['F1'] = [f1]
        metrics['CR'] = [cr]
        metrics['CR_pred_cutoff'] = [cr_p]
#        metrics['PrecisionScore_cutoff_' + str(desired_cutoff * 100)] = [cr_p_des_cutt]
#        metrics['Volumes_Criterion_rate_predicted_' + str(desired_cutoff * 100)] = [cr_p_des_vol]
        metrics['Volumes'] = [vol]
        return metrics, df


    def generate_predictions_and_metrics(self, y_true, df):
        """
        Generate predictions and metrics based on the input data.

        Args:
            y_true: True target values.
            df: Input DataFrame.

        Returns:
            metrics_df: DataFrame with evaluation metrics.
            df: Updated DataFrame with predictions and probabilities.
        """

        df[f'{self.model_name}_y_pred'] = self.model.predict(df[self.final_features])
        df[f'{self.model_name}_deciles_predict'] = pd.qcut(df[f'{self.model_name}_y_pred'], 10, duplicates='drop',
                                                           labels=False)
        if self.model_name == 'xgb':
            df[f'{self.model_name}_y_pred_prob'] = self.model.predict_proba(df[self.model.get_booster().feature_names])[
                                                   :, 1]
        else:
            df[f'{self.model_name}_y_pred_prob'] = self.model.predict_proba(df[self.final_features])[
                                                   :, 1]
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

        ac, auc, prec, recall, f1, cr, cr_p, vol = get_metrics(y_pred=df[f'{self.model_name}_y_pred'], y_true=y_true,
                                                               y_pred_prob=df[f'{self.model_name}_y_pred_prob'])

        desired_cutoff = 0.5
        df_temp_tocalc_proba = df[df[f'{self.model_name}_y_pred_prob'] >= desired_cutoff].copy()
        cr_p_des_cutt = round(df_temp_tocalc_proba[self.params['criterion_column']].sum() / df_temp_tocalc_proba[
            self.params['criterion_column']].count(), 2)
        cr_p_des_vol = round(
            df_temp_tocalc_proba[f'{self.model_name}_y_pred'].sum() / df[self.params['criterion_column']].count(), 2)

        metrics_df = pd.DataFrame()
        metrics_df['AccuracyScore'] = [ac]
        metrics_df['AUC'] = [auc]
        metrics_df['PrecisionScore'] = [prec]
        metrics_df['Recall'] = [recall]
        metrics_df['F1'] = [f1]
        metrics_df['CR'] = [cr]
        metrics_df['CR_pred_cutoff'] = [cr_p]
        metrics_df['PrecisionScore_cutoff_' + str(desired_cutoff * 100)] = [cr_p_des_cutt]
        metrics_df['Volumes_Criterion_rate_predicted_' + str(desired_cutoff * 100)] = [cr_p_des_vol]
        metrics_df['Volumes'] = [vol]
        return metrics_df, df
