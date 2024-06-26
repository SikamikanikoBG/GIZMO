import os
import pickle
import definitions
# import logging # for mlflow debug
use_mlflow = False
try:
    import mlflow
    mlflow.set_tracking_uri(definitions.mlflow_tracking_uri)
    use_mlflow = True
    # logging.getLogger("mlflow").setLevel(logging.DEBUG) # for mlflow debug
    # mlflow.enable_async_logging(enable=True) # debug

except:
    pass

import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.classes.BaseModeller import BaseModeller
from src.classes.SessionManager import SessionManager
from src.functions.data_prep.misc_functions import correlation_matrix
from src.functions.data_prep.raw_features_to_list import raw_features_to_list
from src.functions.printing_and_logging import print_end, print_and_log, print_train


class ModuleClass(SessionManager):
    def __init__(self, args):
        """
        Initializes the ModuleClass object.

        Args:
            args: Arguments passed to the object.

        Returns:
            None
        """

        SessionManager.__init__(self, args)

        if use_mlflow:
            mlflow.set_experiment(definitions.mlflow_prefix + "_" + self.project_name)
        
        self.metrics_df = pd.DataFrame()
        self.is_multiclass = self.loader.in_df[self.params['criterion_column']].nunique() > 2

    def run(self):
        """
        Orchestrates the actions to be performed by this class.

        Returns:
            None
        """

        self.prepare()
        print_train()

        # remove last n rows where there is no enough time/periods to calculate the criterion
        # shape_of_df = len(self.loader.in_df)
        # records_to_keep = shape_of_df - self.args.period
        # self.loader.in_df = self.loader.in_df.head(records_to_keep).copy()

        self.dt_ = ['xgb', 'rf', 'dt']
        models = self.dt_
        if self.is_multiclass:
            models = ['xgb']
            self.encode_criterion_column()        

        self.split_temporal_validation_periods()
        self.split_train_test_df()
        self.create_train_session_folder()

        """
        models should be specified as following:
        xgb for XGBoost
        rf for Random forest
        dt for Decision trees
        lr for Logistic Regression
        """

        mlflow.start_run(run_name=self.session_id)

        for model in models:
            if use_mlflow:
                mlflow.start_run(nested=True, run_name=model)
                mlflow.xgboost.autolog()

                # Get the tracking URI
                tracking_uri = mlflow.get_tracking_uri()
                print("MLflow Tracking URI:", tracking_uri)

                # Get the artifact root URI (if configured)
                artifact_root = mlflow.get_artifact_uri()
                print("MLflow Artifact Root URI:", artifact_root)

            metrics = self.create_model_procedure(model)
            
            if use_mlflow:
                self.log_to_mlflow(metrics)
            
            self.metrics_df = self.metrics_df.append(metrics)
            
            if use_mlflow:
                mlflow.end_run()

        self.save_results()

        print_end()

    def encode_criterion_column(self):
        """
        Encodes the criterion column using LabelEncoder and saves the encoded classes to a CSV file.

        Returns:
            None
        """

        le = LabelEncoder()
        self.loader.in_df_f[self.criterion_column] = le.fit_transform(self.loader.in_df_f[self.criterion_column])
        if self.under_sampling:
            self.loader.in_df[self.criterion_column] = le.transform(self.loader.in_df[self.criterion_column])
            print(self.loader.in_df[self.criterion_column])
        self.encoded_classes = pd.DataFrame(le.classes_) \
            .reset_index() \
            .rename(columns={'index': 'class_encoded_as', 0: 'class_label'}) 
        self.encoded_classes.to_csv(self.output_data_folder_name + self.input_data_project_folder + '/' + 'encoded_labels.csv', index=False)

    def log_to_mlflow(self, metrics: pd.DataFrame):
        '''
        Logs metrics to MLFlow server by DataSet.

        Args:
            metrics (pd.DataFrame): DataFrame containing metrics to be logged.

        Returns:
            None
        '''

        for row in metrics.iterrows():
            mlflow.start_run(nested=True, run_name=f"{row[1]['DataSet']}")            
            for idx, val in row[1].items():
                if type(val) != str:
                    mlflow.log_metric(idx, val)               
            mlflow.end_run()


    def create_model_procedure(self, model_type):
        """
        Creates a model training procedure for the specified model type.

        Args:
            model_type (str): Type of the model to be trained.

        Returns:
            pd.DataFrame: Metrics for the trained model.
        """

        print_and_log(f"Starting training procedure for {model_type}", 'YELLOW')
        globals()['self.modeller_' + model_type] = BaseModeller(model_name=model_type,
                                                                params=self.params,
                                                                final_features=self.loader.final_features,
                                                                cut_offs=self.cut_offs)

        # Debug
        # print(f"@------XGB_CONFIG_CHOSEN_STANDARD.PY------@")
        # print()
        # for k in globals()['self.modeller_' + model_type].params:
        #     print(f"{k}")
        # print(self.params)

        if model_type == 'lr':  # Reminder: lr stands for logistic regression
            pass
        else:
            self.train_modelling_procedure_trees(model_type)

        self.validation_modelling_procedure(model_type)

        pickle.dump(globals()['self.modeller_' + model_type],
                    open(self.session_id_folder + f'/{model_type}/model_train.pkl', 'wb'))
        globals()['self.modeller_' + model_type].results.to_csv(
            self.session_id_folder + f'/{model_type}/feat_importance.csv', index=False)
        print_and_log(f"Feature importance: {globals()['self.modeller_' + model_type].results}", '')
        return globals()['self.modeller_' + model_type].metrics

    def save_results(self):
        """
        Saves the training and testing dataframes, as well as the metrics to files.

        Returns:
            None
        """

        self.loader.train_X.reset_index().to_feather(self.session_id_folder + '/df_x_train.feather')
        self.loader.test_X.reset_index().to_feather(self.session_id_folder + '/df_x_test.feather')
        self.loader.t1df.reset_index().to_feather(self.session_id_folder + '/df_t1df.feather')
        self.loader.t2df.reset_index().to_feather(self.session_id_folder + '/df_t2df.feather')
        self.loader.t3df.reset_index().to_feather(self.session_id_folder + '/df_t3df.feather')
        self.metrics_df.to_csv(self.session_id_folder + '/models.csv', index=False)

    def validation_modelling_procedure(self, model_type):
        """
        Performs the validation modelling procedure for the specified model type.

        Args:
            model_type (str): Type of the model being validated.

        Returns:
            None
        """

        validation_dataframes_name = ['test_X', 't1df', 't2df', 't3df']
        validation_dataframes_X = [self.loader.test_X, self.loader.t1df, self.loader.t2df, self.loader.t3df]
        validation_dataframes_y = [self.loader.y_test, self.loader.t1df_y, self.loader.t2df_y, self.loader.t3df_y]

        count = 0
        if not self.is_multiclass:
            for dataframe in validation_dataframes_X:            
                metrics, validation_dataframes_X[count] = \
                    globals()['self.modeller_' + model_type] \
                        .generate_predictions_and_metrics(
                                    y_true=validation_dataframes_y[count],
                                    df=validation_dataframes_X[count]
                    )            
                globals()['self.modeller_' + model_type].metrics = globals()['self.modeller_' + model_type].metrics.append(
                metrics)
                globals()['self.modeller_' + model_type].metrics['DataSet'].iloc[-1] = validation_dataframes_name[count]
                count += 1
        else: 
            # This loop needs to be unrolled because multiple columns need to be added to the dataframe which is done with 
            # pd.concat, which creates new Dataframe, it gets replaced in the list, but does not update the end loader DF.
            metrics, self.loader.test_X = \
                    globals()['self.modeller_' + model_type] \
                        .generate_multiclass_predictions_and_metrics(
                            self.loader.y_test,
                            self.loader.test_X,
                            self.encoded_classes
                    )
            globals()['self.modeller_' + model_type].metrics = globals()['self.modeller_' + model_type].metrics.append(
                metrics)
            globals()['self.modeller_' + model_type].metrics['DataSet'].iloc[-1] = 'test_X'

            metrics, self.loader.t1df = \
                    globals()['self.modeller_' + model_type] \
                        .generate_multiclass_predictions_and_metrics(
                            self.loader.t1df_y,
                            self.loader.t1df,
                            self.encoded_classes
                    )
            globals()['self.modeller_' + model_type].metrics = globals()['self.modeller_' + model_type].metrics.append(
                metrics)
            globals()['self.modeller_' + model_type].metrics['DataSet'].iloc[-1] = 't1df'

            metrics, self.loader.t2df = \
                    globals()['self.modeller_' + model_type] \
                        .generate_multiclass_predictions_and_metrics(
                            self.loader.t2df_y,
                            self.loader.t2df,
                            self.encoded_classes
                    )
            globals()['self.modeller_' + model_type].metrics = globals()['self.modeller_' + model_type].metrics.append(
                metrics)
            globals()['self.modeller_' + model_type].metrics['DataSet'].iloc[-1] = 't2df'

            metrics, self.loader.t3df = \
                    globals()['self.modeller_' + model_type] \
                        .generate_multiclass_predictions_and_metrics(
                            self.loader.t3df_y,
                            self.loader.t3df,
                            self.encoded_classes
                    )
            globals()['self.modeller_' + model_type].metrics = globals()['self.modeller_' + model_type].metrics.append(
                metrics)
            globals()['self.modeller_' + model_type].metrics['DataSet'].iloc[-1] = 't3df'

        globals()['self.modeller_' + model_type].metrics['Method'] = model_type
        globals()['self.modeller_' + model_type].metrics['NbFeatures'] = len(
            globals()['self.modeller_' + model_type].final_features)

    def train_modelling_procedure_trees(self, model_type):
        """
        This method is used to run the necessary steps and train a model, specifically for tree models.

        Args:
            model_type (str): Type of the model being trained.

        This method performs the following steps:
            1. Loads the model for the specified model type.

            2. Fits the training models procedure.

            3. Calculates feature importances and selects top features based on importance.

            4. Optionally removes specified features from the final feature list.

            5. Fits the model again with the selected final features.

            6. Saves cost function graphs, AUC curves, and error curves.

            7. Generates predictions and metrics for the training data.

            8. Appends the metrics to the model's metrics dataframe.

            9. Calculates correlation matrix for the final features.

        Returns:
            None
        """

        results = pd.DataFrame()

        if globals()['self.modeller_' + model_type].trees_features_to_include:
            globals()['self.modeller_' + model_type].final_features = globals()[
                'self.modeller_' + model_type].trees_features_to_include

        # load model
        globals()['self.modeller_' + model_type].load_model(self.is_multiclass)

        self.training_models_fit_procedure(model_type)

        results['columns'] = self.loader.train_X[globals()['self.modeller_' + model_type].final_features].columns
        results['importances'] = globals()['self.modeller_' + model_type].model.feature_importances_
        results.sort_values(by='importances', ascending=False, inplace=True)
        print("@_______DEBUG________@")
        print(f"results: {results}")
        print(f"results: {results.shape}")

        # Select importances between 0 and 0.95 and keep top args.nb_tree_features features
        #results = results[results['importances'] > 0]
        #results = results[results['importances'] < 0.95]

        if self.args.nb_tree_features:
            nb_features = int(self.args.nb_tree_features)
            results = results[:nb_features]
        else:
            results = results[:100]
        globals()['self.modeller_' + model_type].final_features = results['columns'].unique().tolist()

        features_removed = False
        if globals()['self.modeller_' + model_type].trees_features_to_exclude:
            print_and_log(
                f'Removing features specified in the params file: {globals()["self.modeller_" + model_type].trees_features_to_exclude}',
                'YELLOW')
            for el in globals()['self.modeller_' + model_type].trees_features_to_exclude:
                try:
                    globals()['self.modeller_' + model_type].final_features.remove(el)
                    results = results[~results['columns'].str.contains(el)]
                    features_removed = True
                except:
                    pass
        
        # fitting new model only with selected final features       
        # TODO if features_removed:
        self.training_models_fit_procedure(model_type)

        try:
            # Saving cost function graphs:
            # retrieve performance metrics
            results_evals = globals()['self.modeller_' + model_type].model.evals_result()
            print(results)
            # plot learning curves
            pyplot.plot(results_evals['validation_0']['mlogloss' if self.is_multiclass else 'logloss'], label='train')
            pyplot.plot(results_evals['validation_1']['mlogloss' if self.is_multiclass else 'logloss'], label='test')
            # show the legend
            pyplot.legend()
            pyplot.savefig(f'{self.session_id_folder}/cost_graph_{model_type}.png')
            mlflow.log_artifact(f'{self.session_id_folder}/cost_graph_{model_type}.png', 'graphs')
            try:
                pyplot.savefig(f'{self.log_folder_name}/cost_graph_{model_type}_{len(self.loader.train_X)}_{self.session_id}.png')
            except Exception as e:
                print_and_log(f"{e}", "YELLOW")
                pass
            pyplot. clf()
            
            
            # plot auc curves @debug
            # pyplot.plot(results_evals['validation_0']['auc'], label='train')
            # pyplot.plot(results_evals['validation_1']['auc'], label='test')

            # show the legend
            pyplot.legend()
            # show the plot
            # pyplot.show()
            pyplot.savefig(f'{self.session_id_folder}/auc_graph_{model_type}.png')
            mlflow.log_artifact(f'{self.session_id_folder}/auc_graph_{model_type}.png', 'graphs')
            try:
                pyplot.savefig(f'{self.log_folder_name}/auc_graph_{model_type}_{len(self.loader.train_X)}_{self.session_id}.png')
            except Exception as e:
                print_and_log(f"{e}", "YELLOW")
                pass
            pyplot. clf()
            
            # plot error curves
            pyplot.plot(results_evals['validation_0']['merror' if self.is_multiclass else 'error'], label='train')
            pyplot.plot(results_evals['validation_1']['merror' if self.is_multiclass else 'error'], label='test')
            # show the legend
            pyplot.legend()
            # show the plot
            # pyplot.show()
            pyplot.savefig(f'{self.session_id_folder}/error_graph_{model_type}.png')
            mlflow.log_artifact(f'{self.session_id_folder}/error_graph_{model_type}.png', 'graphs')
            try:
                pyplot.savefig(f'{self.log_folder_name}/error_graph_{model_type}_{len(self.loader.train_X)}_{self.session_id}.png')
            except Exception as e:
                print_and_log(f"{e}", "YELLOW")
                pass
        except Exception as e:
            print_and_log(f"EVAL Validation_1: {e}", "YELLOW")
            pass

        if self.is_multiclass:            
            metrics, self.loader.train_X = globals()['self.modeller_' + model_type].generate_multiclass_predictions_and_metrics(
                self.loader.y_train, self.loader.train_X, self.encoded_classes
            )
        else:
            metrics, self.loader.train_X = globals()[
                'self.modeller_' + model_type].generate_predictions_and_metrics(
                y_true=self.loader.y_train,
                df=self.loader.train_X)

        globals()['self.modeller_' + model_type].metrics = globals()['self.modeller_' + model_type].metrics.append(
            metrics)
        globals()['self.modeller_' + model_type].metrics['DataSet'] = 'train_X'
        globals()['self.modeller_' + model_type].results = results

        correlation_matrix(X=self.loader.train_X[globals()['self.modeller_' + model_type].final_features],
                           y=None,
                           flag_matrix='all',
                           input_data_project_folder=None,
                           session_id_folder=self.session_id_folder,
                           model_corr=model_type,
                           flag_raw='', keep_cols=None)
        globals()['self.modeller_' + model_type].raw_features = raw_features_to_list(
            globals()['self.modeller_' + model_type].final_features)
        try:
            correlation_matrix(X=self.loader.train_X[globals()['self.modeller_' + model_type].raw_features],
                           y=None,
                           flag_matrix='all',
                           input_data_project_folder=None,
                           session_id_folder=self.session_id_folder,
                           model_corr=model_type,
                           flag_raw='yes', keep_cols=None)
        except Exception as e:
            print_and_log(f"[ TRAINING ] Raw featires correlation matrix error for model: {model_type}. Error: {e}", "RED")

    def training_models_fit_procedure(self, model_type):
        """
        This method is used to fit the model for the specified model type.

        Args:
            model_type (str): Type of the model being trained.

        This method performs the following steps:
            Fits the model with the training data and evaluates on the test data.

        Returns:
            None
        """
        # Debug print statements:

        print("@-----------------DEBUG PRINT OF FITTING PROCEDURE-----------------@")
        x_train = self.loader.train_X_us[globals()['self.modeller_' + model_type].final_features]
        y_train = self.loader.y_train_us
        print(f"x_train: {x_train.shape}")
        print(f"y_train: {y_train.shape}")

        x_test = self.loader.test_X_us[globals()['self.modeller_' + model_type].final_features]
        y_test = self.loader.y_test_us
        print(f"x_test: {x_test.shape}")
        print(f"y_test: {y_test.shape}")

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
        """
        Creates a session folder for the training session.

        Returns:
            None
        """

        print_and_log('Creating session folder. Starting', 'YELLOW')
        self.session_id = 'TRAIN_' + self.input_data_project_folder + '_' + str(self.start_time) + '_' + self.tag
        self.session_id_folder = self.session_folder_name + self.session_id
        os.mkdir(self.session_id_folder)
        os.mkdir(self.session_id_folder + '/dt/')
        os.mkdir(self.session_id_folder + '/rf/')
        os.mkdir(self.session_id_folder + '/xgb/')
        os.mkdir(self.session_id_folder + '/lr/')
        print_and_log('Creating session folder. Done', '')

    def split_train_test_df(self):
        """
        Splits the training and testing dataframes, as well as the temporal validation dataframes.

        Returns:
            None
        """
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
        """
        Splits the temporal validation dataframes based on the specified periods.

        Returns:
            None
        """

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
            ~self.loader.in_df[self.observation_date_column].isin([self.t1df_period, self.t2df_period, self.t3df_period])].copy()

        if self.under_sampling:
            self.loader.in_df_f = self.loader.in_df_f[
                ~self.loader.in_df_f[self.observation_date_column].isin([self.t1df_period, self.t2df_period, self.t3df_period])].copy()            

        self.loader.t1df_y = self.loader.t1df[self.criterion_column]
        self.loader.t2df_y = self.loader.t2df[self.criterion_column]
        self.loader.t3df_y = self.loader.t3df[self.criterion_column]

        cr_t1 = self.loader.t1df[self.criterion_column].value_counts(dropna=False, normalize=True)
        cr_t2 = self.loader.t2df[self.criterion_column].value_counts(dropna=False, normalize=True)
        cr_t3 = self.loader.t3df[self.criterion_column].value_counts(dropna=False, normalize=True)

        print_and_log(f"[ TEMPORAL VALIDATION ] CR(%):\nt1\n{cr_t1},\nt2\n{cr_t2},\nt3\n{cr_t3}.", "")

        if len(self.loader.t1df) == 0 or len(self.loader.t2df) == 0 or len(self.loader.t3df) == 0:
            print_and_log(
                f"[ TEMPORAL VALIDATION ] ERROR: data for some of the temp validation periods is missing. Termination",
                "RED")
            quit()
        print_and_log("Splitting temporal validation dataframes. Done", '')
