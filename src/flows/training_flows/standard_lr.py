import os
import pickle
from multiprocessing import Pool

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classes.SessionManager import SessionManager
from src.classes.BaseModeller import BaseModeller
from src.functions.data_prep.misc_functions import correlation_matrix
from src.functions.printing_and_logging import print_end, print_and_log, print_train
from src.functions.data_prep.raw_features_to_list import raw_features_to_list


class ModuleClass(SessionManager):
    def __init__(self, args):
        SessionManager.__init__(self, args)
        self.metrics_df = pd.DataFrame()

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        self.prepare()
        print_train()

        self.split_temporal_validation_periods()
        self.split_train_test_df()
        self.create_train_session_folder()

        models = ['xgb', 'rf', 'dt', 'lr']
        pool = Pool(5)
        metrics = pool.map(self.create_model_procedure, models)
        pool.close()
        pool.join()
        self.metrics_df = self.metrics_df.append(metrics)

        print(self.metrics_df.head(50))
        self.save_results()

        print_end()
        self.run_time_calc()

    def create_model_procedure(self, model_type):
        print_and_log(f"Starting training procedure for {model_type}", 'YELLOW')
        globals()['self.modeller_' + model_type] = BaseModeller(model_name=model_type,
                                                                params=self.params,
                                                                final_features=self.loader.final_features,
                                                                cut_offs=self.cut_offs)

        self.train_modelling_procedure_trees(model_type)
        self.validation_modelling_procedure(model_type)

        pickle.dump(globals()['self.modeller_' + model_type].model,
                    open(self.session_id_folder + f'/{model_type}/model_train.pkl', 'wb'))
        globals()['self.modeller_' + model_type].results.to_csv(
            self.session_id_folder + f'/{model_type}/feat_importance.csv', index=False)
        print_and_log(f"Feature importance: {globals()['self.modeller_' + model_type].results}", '')
        return self.metrics_df.append(globals()['self.modeller_' + model_type].metrics)

    def save_results(self):
        self.loader.train_X.reset_index().to_feather(self.session_id_folder + '/df_x_train.feather')
        self.loader.test_X.reset_index().to_feather(self.session_id_folder + '/df_x_test.feather')
        self.loader.t1df.reset_index().to_feather(self.session_id_folder + '/df_t1df.feather')
        self.loader.t2df.reset_index().to_feather(self.session_id_folder + '/df_t2df.feather')
        self.loader.t3df.reset_index().to_feather(self.session_id_folder + '/df_t3df.feather')
        self.metrics_df.to_csv(self.session_id_folder + '/models.csv', index=False)

    def validation_modelling_procedure(self, model_type):
        validation_dataframes_name = ['test_X', 't1df', 't2df', 't3df']
        validation_dataframes_X = [self.loader.test_X, self.loader.t1df, self.loader.t2df, self.loader.t3df]
        validation_dataframes_y = [self.loader.y_test, self.loader.t1df_y, self.loader.t2df_y, self.loader.t3df_y]

        count = 0
        for dataframe in validation_dataframes_X:
            globals()['self.modeller_' + model_type].metrics, validation_dataframes_X[count] = globals()['self.modeller_' + model_type].metrics.append(
                globals()['self.modeller_' + model_type].generate_predictions_and_metrics(
                    y_true=validation_dataframes_y[count],
                    df=validation_dataframes_X[count]))
            globals()['self.modeller_' + model_type].metrics['DataSet'].iloc[-1] = validation_dataframes_name[count]
            count += 1

        globals()['self.modeller_' + model_type].metrics['Method'] = model_type
        globals()['self.modeller_' + model_type].metrics['NbFeatures'] = len(
            globals()['self.modeller_' + model_type].final_features)

    def train_modelling_procedure_trees(self, model_type):
        """
        This method is used in order to run the needed steps and train a model. It is used for tree models since
        their methods are the same.
        Args:
            model_type: Needed in order to distinguish slight differences in the training procedure

        Returns:

        """
        results = pd.DataFrame()

        if model_type == 'lr':
            if globals()['self.modeller_' + model_type].lr_features_to_include:
                globals()['self.modeller_' + model_type].final_features.append(globals()[
                    'self.modeller_' + model_type].lr_features_to_include)
        else:
            if globals()['self.modeller_' + model_type].trees_features_to_include:
                globals()['self.modeller_' + model_type].final_features = globals()[
                    'self.modeller_' + model_type].trees_features_to_include

        # load model
        globals()['self.modeller_' + model_type].load_model()

        if model_type == 'lr':
            self.training_models_fit_procedure_lr()
        else:
            self.training_models_fit_procedure(model_type)

        results['columns'] = self.loader.train_X[globals()['self.modeller_' + model_type].final_features].columns
        results['importances'] = globals()['self.modeller_' + model_type].model.feature_importances_
        results.sort_values(by='importances', ascending=False, inplace=True)

        # Select importances between 0 and 0.95 and keep top 30 features
        results = results[results['importances'] > 0]
        results = results[results['importances'] < 0.95]
        results = results[:30]
        globals()['self.modeller_' + model_type].final_features = results['columns'].unique().tolist()

        if globals()['self.modeller_' + model_type].trees_features_to_exclude:
            print_and_log(
                f'Removing features specified in the params file: {globals()["self.modeller_" + model_type].trees_features_to_exclude}',
                'YELLOW')
            for el in globals()['self.modeller_' + model_type].trees_features_to_exclude:
                try:
                    globals()['self.modeller_' + model_type].final_features.remove(el)
                    results = results[~results['columns'].str.contains(el)]
                except:
                    pass

        # fitting new model only with selected final features
        self.training_models_fit_procedure(model_type)

        print_and_log(f'FINAL FEATURES: {globals()["self.modeller_" + model_type].final_features}', 'GREEN')

        globals()['self.modeller_' + model_type].metrics, self.loader.train_X = globals()['self.modeller_' + model_type].metrics.append(
            globals()[
                'self.modeller_' + model_type].generate_predictions_and_metrics(
                y_true=self.loader.y_train,
                df=self.loader.train_X))
        globals()['self.modeller_' + model_type].metrics['DataSet'] = 'train_X'
        globals()['self.modeller_' + model_type].results = results

        correlation_matrix(X=self.loader.train_X[globals()['self.modeller_' + model_type].final_features],
                           y=None,
                           flag_matrix='all',
                           input_data_project_folder=None,
                           session_id_folder=self.session_id_folder,
                           model_corr=model_type,
                           flag_raw='')
        globals()['self.modeller_' + model_type].raw_features = raw_features_to_list(
            globals()['self.modeller_' + model_type].final_features)
        correlation_matrix(X=self.loader.train_X[globals()['self.modeller_' + model_type].raw_features],
                           y=None,
                           flag_matrix='all',
                           input_data_project_folder=None,
                           session_id_folder=self.session_id_folder,
                           model_corr=model_type,
                           flag_raw='yes')

    def training_models_fit_procedure(self, model_type):
        if self.under_sampling:
            # print_and_log('\n\t *** UNDERSAMPLING MODEL ***', 'YELLOW')
            globals()['self.modeller_' + model_type].model_fit(
                self.loader.train_X_us[globals()['self.modeller_' + model_type].final_features],
                self.loader.y_train_us,
                self.loader.test_X_us[globals()['self.modeller_' + model_type].final_features], self.loader.y_test_us)
        else:
            globals()['self.modeller_' + model_type].model_fit(
                self.loader.train_X[globals()['self.modeller_' + model_type].final_features],
                self.loader.y_train,
                self.loader.test_X[globals()['self.modeller_' + model_type].final_features], self.loader.y_test)

    def create_train_session_folder(self):
        print_and_log('Createing session folder. Starting', 'YELLOW')
        self.session_id = 'TRAIN_' + self.input_data_project_folder + '_' + str(self.start) + '_' + self.tag
        self.session_id_folder = self.session_folder_name + self.session_id
        os.mkdir(self.session_id_folder)
        os.mkdir(self.session_id_folder + '/dt/')
        os.mkdir(self.session_id_folder + '/rf/')
        os.mkdir(self.session_id_folder + '/xgb/')
        os.mkdir(self.session_id_folder + '/lr/')
        print_and_log('Createing session folder. Done', '')

    def split_train_test_df(self):
        print_and_log("Splitting train and test dataframes. Starting", '')
        if self.under_sampling:
            self.loader.train_X, self.loader.test_X, self.loader.y_train, self.loader.y_test = train_test_split(
                self.loader.in_df_f, self.loader.in_df_f[self.criterion_column], test_size=0.33,
                random_state=42)

            self.loader.train_X_us, self.loader.test_X_us, self.loader.y_train_us, self.loader.y_test_us = train_test_split(
                self.loader.in_df, self.loader.in_df[self.criterion_column], test_size=0.33, random_state=42)
            del self.loader.in_df
            del self.loader.in_df_f
        else:
            self.loader.train_X, self.loader.test_X, self.loader.y_train, self.loader.y_test = train_test_split(
                self.loader.in_df, self.loader.in_df[self.criterion_column], test_size=0.33, random_state=42)
            del self.loader.in_df
        print_and_log('Splitting train and test dataframes. Done', '')

    def split_temporal_validation_periods(self):
        print_and_log(
            f'\t All observation dates before splitting the '
            f'file: {self.observation_date_column}: {self.loader.in_df[self.observation_date_column].unique()}', '')
        if self.under_sampling:
            print_and_log(f"Splitting temporal validation dataframes"
                          f", {self.t1df_period}, {self.t2df_period}, {self.t3df_period}", '')
            self.loader.t1df = self.loader.in_df_f[
                self.loader.in_df_f[self.observation_date_column] == self.t1df_period].copy()
            print_and_log(f"t1 done. Shape: {len(self.loader.t1df)}", '')

            self.loader.t2df = self.loader.in_df_f[
                self.loader.in_df_f[self.observation_date_column] == self.t2df_period].copy()
            print_and_log(f"t2 done. Shape: {len(self.loader.t2df)}", '')

            self.loader.t3df = self.loader.in_df_f[
                self.loader.in_df_f[self.observation_date_column] == self.t3df_period].copy()
            print_and_log(f"t3 done. Shape: {len(self.loader.t3df)}", '')
        else:
            print_and_log(
                f"Splitting temporal validation dataframes, {self.t1df_period}, {self.t2df_period}, {self.t3df_period}",
                '')
            self.loader.t1df = self.loader.in_df[
                self.loader.in_df[self.observation_date_column] == self.t1df_period].copy()
            print_and_log(f"t1 done. Shape: {len(self.loader.t1df)}", '')

            self.loader.t2df = self.loader.in_df[
                self.loader.in_df[self.observation_date_column] == self.t2df_period].copy()
            print_and_log(f"t2 done. Shape: {len(self.loader.t2df)}", '')

            self.loader.t3df = self.loader.in_df[
                self.loader.in_df[self.observation_date_column] == self.t3df_period].copy()
            print_and_log(f"t3 done. Shape: {len(self.loader.t3df)}", '')

        self.loader.in_df = self.loader.in_df[
            self.loader.in_df[self.observation_date_column] != self.t1df_period].copy()
        self.loader.in_df = self.loader.in_df[
            self.loader.in_df[self.observation_date_column] != self.t2df_period].copy()
        self.loader.in_df = self.loader.in_df[
            self.loader.in_df[self.observation_date_column] != self.t3df_period].copy()

        if self.params['under_sampling']:
            self.loader.in_df_f = self.loader.in_df_f[
                self.loader.in_df_f[self.observation_date_column] != self.t1df_period].copy()
            self.loader.in_df_f = self.loader.in_df_f[
                self.loader.in_df_f[self.observation_date_column] != self.t2df_period].copy()
            self.loader.in_df_f = self.loader.in_df_f[
                self.loader.in_df_f[self.observation_date_column] != self.t3df_period].copy()

        self.loader.t1df_y = self.loader.t1df[self.criterion_column]
        self.loader.t2df_y = self.loader.t2df[self.criterion_column]
        self.loader.t3df_y = self.loader.t3df[self.criterion_column]
        print_and_log("Splitting temporal validation dataframes. Done", '')

    def training_models_fit_procedure_lr(self):
        """model_type = 'lr'
        globals()['self.modeller_' + model_type].model_fit(self.loader.train_X_us[globals()['self.modeller_' + model_type].final_features],
                self.loader.y_train_us,
                self.loader.test_X_us[globals()['self.modeller_' + model_type].final_features], self.loader.y_test_us)

        # todo: create training method for lr to be able to do this loop of modelling. BUT with US going to lunch
        Coefficients = pd.read_html(globals()['self.modeller_' + model_type].lr_table.tables[1].as_html(), header=0, index_col=0)[0]
        Coefficients = Coefficients.reset_index()
        Coefficients = Coefficients[['index', 'coef', 'P>|z|']]
        Coefficients = Coefficients.rename(columns={'P>|z|': 'error'})
        Coefficients = Coefficients[Coefficients.error < 0.01]
        Coefficients = Coefficients['index'].tolist()
        Coefficients.remove('const')

        globals()['self.modeller_' + model_type].model_fit(
            self.loader.train_X_us[Coefficients],
            self.loader.y_train_us,
            self.loader.test_X_us[globals()['self.modeller_' + model_type].final_features], self.loader.y_test_us)
        Coefficients = pd.read_html(table.tables[1].as_html(), header=0, index_col=0)[0]
        Coefficients = Coefficients.reset_index()
        Coefficients = Coefficients[['index', 'coef', 'P>|z|']]
        Coefficients = Coefficients.rename(columns={'P>|z|': 'error'})
        Coefficients = Coefficients[Coefficients.error < 0.01]
        Coefficients['coef_abs'] = abs(Coefficients['coef'])
        Coefficients = Coefficients[Coefficients.coef_abs > 0.00005]
        Coefficients = Coefficients['index'].tolist()
        Coefficients.remove('const')

        lr_model, table, train_auc = lr(df_sample[Coefficients], criterion)
        Coefficients = pd.read_html(table.tables[1].as_html(), header=0, index_col=0)[0]
        Coefficients = Coefficients.reset_index()
        Coefficients = Coefficients[['index', 'coef', 'P>|z|']]
        Coefficients = Coefficients.rename(columns={'P>|z|': 'error'})
        Coefficients = Coefficients['index'].tolist()
        Coefficients.remove('const')
        if len(Coefficients) > 8:
            df_abs = abs(df[Coefficients].corr())
            df_abs = df_abs.where(df_abs < 1)
            df_abs = df_abs.where(df_abs > 0.5)
            df_abs = df_abs.isnull().values.all()
            if df_abs:
                if train_auc > auc_max:
                    tries = 0
                    auc_max = train_auc
                print(
                    '\t After {} attempts: LR Model found with AUC: {} and {} nb of features. AUC max so far {}'.format(
                        tries, round(train_auc, 2), len(Coefficients), round(auc_max, 2)))
                results_lr = results_lr.append(
                    {'auc': train_auc, 'features': Coefficients},
                    ignore_index=True)

        except:
        pass"""


        """final_features = results_lr[results_lr['auc'] == results_lr['auc'].max()]
        final_features = final_features['features'].iloc[0]
        print(len(final_features), final_features)
        logging.info(f'The number of features is {len(final_features)} for feature {final_features}')
        
        lr_model, table, train_auc = lr(df[final_features], criterion)
        print('\t WINNER AUC: {}'.format(train_auc))
        logging.info(f'WINNER AUC: {train_auc}')
        
        lr_table = pd.read_html(table.tables[1].as_html(), header=0, index_col=0)[0]
        lr_table = lr_table.reset_index()
        lr_table = lr_table[['index', 'coef', 'P>|z|']]
        lr_table = lr_table.rename(columns={'P>|z|': 'error'})
        lr_model.feature_names = final_features
        
        else:
        lr_model = model_to_predict
        final_features = lr_model.feature_names
        
        df = sm.add_constant(df, has_constant='add')  # add constant
        final_features.insert(0, 'const')
        df['lr_y_pred'] = lr_model.predict(df[final_features])
        df['lr_y_pred_prob'] = lr_model.predict(df[final_features])
        y_pred = lr_model.predict(df[final_features])
        
        ac, auc, prec, recall, f1 = get_metrics(y_pred=y_pred, y_true=criterion, y_pred_prob=df['lr_y_pred_prob'])
        
        if cut_offs["lr"]:
            df['lr_bands_predict'] = pd.cut(df['lr_y_pred'], bins=cut_offs["lr"], include_lowest=True).astype('str')
            df['lr_bands_predict_proba'] = pd.cut(df['lr_y_pred_prob'], bins=cut_offs["lr"], include_lowest=True).astype(
                'str')
        else:
            df['lr_bands_predict'], _ = cut_into_bands(X=df[['lr_y_pred']], y=criterion, depth=3)
            df['lr_bands_predict_proba'], _ = cut_into_bands(X=df[['lr_y_pred_prob']], y=criterion, depth=3)
        logging.info('LR: Model found')
        
        final_features.remove('const')"""
