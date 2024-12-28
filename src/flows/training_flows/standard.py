import os
import pickle
import definitions
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

use_mlflow = False
try:
    import mlflow
    # import logging
    mlflow.set_tracking_uri(definitions.mlflow_tracking_uri)
    use_mlflow = True
    # logging.getLogger("mlflow").setLevel(logging.DEBUG) # for mlflow debug
    # mlflow.enable_async_logging(enable=True) # debug

except:
    pass

import pandas as pd
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
        Handles both MLflow and non-MLflow execution paths.
        """
        try:
            self.prepare()
            print_train()

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
            # MLflow handling
            mlflow_run = None
            if use_mlflow:
                mlflow_run = mlflow.start_run(run_name=self.session_id)

            try:
                for model in models:
                    if use_mlflow:
                        with mlflow.start_run(nested=True, run_name=model):
                            mlflow.xgboost.autolog()
                            metrics = self.create_model_procedure(model)
                            self.log_to_mlflow(metrics)
                    else:
                        metrics = self.create_model_procedure(model)
                    
                    self.metrics_df = self.metrics_df.append(metrics)

                self.save_results()
                print_end()

            finally:
                # Ensure MLflow run is ended if it was started
                if use_mlflow and mlflow_run:
                    mlflow.end_run()
                    
        except Exception as e:
            print_and_log(f"Error during execution: {str(e)}", "RED")
            if use_mlflow and mlflow_run:
                mlflow.end_run()
            raise
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
        
        # Initialize model
        globals()['self.modeller_' + model_type] = BaseModeller(
            model_name=model_type,
            params=self.params,
            final_features=self.loader.final_features,
            cut_offs=self.cut_offs
        )

        # Train model based on type
        if model_type == 'lr':
            pass
        else:
            self.train_modelling_procedure_trees(model_type)

        # Run validation
        self.validation_modelling_procedure(model_type)

        # Save model artifacts in organized structure
        model_dir = os.path.join(self.session_id_folder, model_type)
        
        # Save trained model
        models_dir = os.path.join(model_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        pickle.dump(
            globals()['self.modeller_' + model_type],
            open(os.path.join(models_dir, 'model_train.pkl'), 'wb')
        )
        
        # Save feature importance
        metrics_dir = os.path.join(model_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        globals()['self.modeller_' + model_type].results.to_csv(
            os.path.join(metrics_dir, 'feature_importance.csv'), 
            index=False
        )
        
        print_and_log(f"Feature importance: {globals()['self.modeller_' + model_type].results}", '')
        return globals()['self.modeller_' + model_type].metrics

    def save_results(self):
        """
        Saves training results in the organized session folder structure.
        
        Saves:
        - Training and test datasets
        - Temporal validation datasets
        - Model metrics
        - Dataset splits
        """
        # Create data directory in session folder if it doesn't exist
        data_dir = os.path.join(self.session_id_folder, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save training and testing dataframes
        self.loader.train_X.reset_index().to_feather(os.path.join(data_dir, 'df_x_train.feather'))
        self.loader.test_X.reset_index().to_feather(os.path.join(data_dir, 'df_x_test.feather'))
        
        # Save temporal validation dataframes
        self.loader.t1df.reset_index().to_feather(os.path.join(data_dir, 'df_t1df.feather'))
        self.loader.t2df.reset_index().to_feather(os.path.join(data_dir, 'df_t2df.feather'))
        self.loader.t3df.reset_index().to_feather(os.path.join(data_dir, 'df_t3df.feather'))
        
        # Save model metrics in the metrics directory
        metrics_dir = os.path.join(self.session_id_folder, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_df.to_csv(os.path.join(metrics_dir, 'model_metrics.csv'), index=False)

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
        Handles the complete training procedure for tree-based models (XGBoost, Random Forest, Decision Tree).
        
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

        # Check if we have predefined features to include
        if globals()['self.modeller_' + model_type].trees_features_to_include:
            print_and_log("Using predefined features from configuration", '')
            globals()['self.modeller_' + model_type].final_features = globals()[
                'self.modeller_' + model_type].trees_features_to_include

        # Load and initialize model
        globals()['self.modeller_' + model_type].load_model(self.is_multiclass)
        print_and_log(f"[ TRAINING ] Initial model fitting for {model_type.upper()}", '')

        # Initial model training
        self.training_models_fit_procedure(model_type)

        # Extract feature importances
        print_and_log("[ TRAINING ] Calculating feature importances", '')
        results['columns'] = self.loader.train_X[globals()['self.modeller_' + model_type].final_features].columns
        results['importances'] = globals()['self.modeller_' + model_type].model.feature_importances_
        results.sort_values(by='importances', ascending=False, inplace=True)

        # Filter features based on importance thresholds
        results = results[results['importances'] > 0]  # Remove zero importance features
        results = results[results['importances'] < 0.95]  # Remove potentially problematic features

        # Select top features
        if self.args.nb_tree_features:
            nb_features = int(self.args.nb_tree_features)
            print_and_log(f"[ TRAINING ] Selecting top {nb_features} features", '')
            results = results[:nb_features]
        else:
            print_and_log(f"[ TRAINING ] Selecting top {definitions.max_features} features", '')
            results = results[:definitions.max_features]

        # Update final features list
        globals()['self.modeller_' + model_type].final_features = results['columns'].unique().tolist()

        # Remove excluded features if specified
        if globals()['self.modeller_' + model_type].trees_features_to_exclude:
            excluded_features = globals()['self.modeller_' + model_type].trees_features_to_exclude
            print_and_log(f'[ TRAINING ] Removing excluded features: {excluded_features}', 'YELLOW')
            
            for feature in excluded_features:
                try:
                    if feature in globals()['self.modeller_' + model_type].final_features:
                        globals()['self.modeller_' + model_type].final_features.remove(feature)
                        results = results[~results['columns'].str.contains(feature)]
                        print_and_log(f"Removed feature: {feature}", '')
                except Exception as e:
                    print_and_log(f"Error removing feature {feature}: {str(e)}", "YELLOW")

        # Final model training with selected features
        print_and_log("[ TRAINING ] Final model fitting with selected features", '')
        self.training_models_fit_procedure(model_type)

        # Generate evaluation plots
        try:
            self.save_evaluation_plots(model_type)
        except Exception as e:
            print_and_log(f"[ TRAINING ] Warning: Could not generate evaluation plots: {str(e)}", "YELLOW")

        # Generate predictions and metrics
        print_and_log("[ TRAINING ] Generating predictions and metrics", '')
        if self.is_multiclass:            
            metrics, self.loader.train_X = globals()['self.modeller_' + model_type].generate_multiclass_predictions_and_metrics(
                self.loader.y_train, 
                self.loader.train_X, 
                self.encoded_classes
            )
        else:
            metrics, self.loader.train_X = globals()['self.modeller_' + model_type].generate_predictions_and_metrics(
                y_true=self.loader.y_train,
                df=self.loader.train_X
            )

        # Update metrics
        globals()['self.modeller_' + model_type].metrics = globals()['self.modeller_' + model_type].metrics.append(metrics)
        globals()['self.modeller_' + model_type].metrics['DataSet'] = 'train_X'
        globals()['self.modeller_' + model_type].results = results

        # Generate correlation matrices
        print_and_log("[ TRAINING ] Generating correlation matrices", '')
        try:
            correlation_matrix(
                X=self.loader.train_X[globals()['self.modeller_' + model_type].final_features],
                y=None,
                flag_matrix='all',
                input_data_project_folder=None,
                session_id_folder=self.session_id_folder,
                model_corr=model_type,
                flag_raw='',
                keep_cols=None
            )

            # Extract and analyze raw features
            globals()['self.modeller_' + model_type].raw_features = raw_features_to_list(
                globals()['self.modeller_' + model_type].final_features
            )
            
            # Generate correlation matrix for raw features
            correlation_matrix(
                X=self.loader.train_X[globals()['self.modeller_' + model_type].raw_features],
                y=None,
                flag_matrix='all',
                input_data_project_folder=None,
                session_id_folder=self.session_id_folder,
                model_corr=model_type,
                flag_raw='yes',
                keep_cols=None
            )
        except Exception as e:
            print_and_log(f"[ TRAINING ] Warning: Error generating correlation matrices: {str(e)}", "RED")

        # Final feature importance report
        n_features = len(globals()['self.modeller_' + model_type].final_features)
        print_and_log(f"[ TRAINING ] Training completed for {model_type.upper()} with {n_features} features", 'GREEN')
        print_and_log(f"Top 10 features by importance:\n{results.head(10)}", '')
    
    def save_evaluation_plots(self, model_type):
        """
        Saves evaluation plots for different model types with appropriate visualizations.
        
        Args:
            model_type (str): The type of model ('xgb', 'rf', or 'dt')
        """
        # Get the plots directory for this model
        plots_dir = os.path.join(self.session_id_folder, model_type, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            model = globals()['self.modeller_' + model_type].model
            
            # For XGBoost models
            if model_type == 'xgb':
                try:
                    eval_results = model.evals_result()
                    
                    # Plot learning curves
                    metric = 'mlogloss' if self.is_multiclass else 'logloss'
                    plt.figure(figsize=(10, 6))
                    plt.plot(eval_results['validation_0'][metric], label='train')
                    plt.plot(eval_results['validation_1'][metric], label='test')
                    plt.title(f'XGBoost Learning Curves - {metric}')
                    plt.xlabel('Iterations')
                    plt.ylabel(metric)
                    plt.legend()
                    plt.savefig(os.path.join(plots_dir, 'learning_curves.png'))
                    plt.close()
                    
                    # Plot error curves
                    metric = 'merror' if self.is_multiclass else 'error'
                    plt.figure(figsize=(10, 6))
                    plt.plot(eval_results['validation_0'][metric], label='train')
                    plt.plot(eval_results['validation_1'][metric], label='test')
                    plt.title(f'XGBoost Error Curves - {metric}')
                    plt.xlabel('Iterations')
                    plt.ylabel(metric)
                    plt.legend()
                    plt.savefig(os.path.join(plots_dir, 'error_curves.png'))
                    plt.close()
                    
                except Exception as e:
                    print_and_log(f"Could not generate XGBoost evaluation plots: {str(e)}", "YELLOW")
            
            # For RF and DT models
            if not self.is_multiclass:
                # Get predictions for ROC curve
                train_probs = model.predict_proba(
                    self.loader.train_X[globals()['self.modeller_' + model_type].final_features]
                )[:, 1]
                test_probs = model.predict_proba(
                    self.loader.test_X[globals()['self.modeller_' + model_type].final_features]
                )[:, 1]
                
                # Calculate ROC curves
                train_fpr, train_tpr, _ = roc_curve(self.loader.y_train, train_probs)
                test_fpr, test_tpr, _ = roc_curve(self.loader.y_test, test_probs)
                
                # Plot ROC curves
                plt.figure(figsize=(10, 6))
                plt.plot(train_fpr, train_tpr, label=f'Train (AUC = {auc(train_fpr, train_tpr):.3f})')
                plt.plot(test_fpr, test_tpr, label=f'Test (AUC = {auc(test_fpr, test_tpr):.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f'{model_type.upper()} ROC Curves')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.savefig(os.path.join(plots_dir, 'roc_curves.png'))
                plt.close()
                
                # Feature importance plot
                feature_importance = pd.DataFrame({
                    'feature': globals()['self.modeller_' + model_type].final_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                # Plot top 20 features
                plt.figure(figsize=(12, 8))
                n_features = min(20, len(feature_importance))
                plt.barh(range(n_features), 
                        feature_importance['importance'].tail(n_features),
                        align='center')
                plt.yticks(range(n_features), 
                        feature_importance['feature'].tail(n_features))
                plt.title(f'{model_type.upper()} Top {n_features} Feature Importance')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
                plt.close()
            
                # Save copies to logs folder if needed
                try:
                    log_plots_dir = os.path.join(self.log_folder_name, 'plots')
                    os.makedirs(log_plots_dir, exist_ok=True)
                    
                    for plot_name in ['learning_curves.png', 'error_curves.png', 'roc_curves.png', 'feature_importance.png']:
                        source = os.path.join(plots_dir, plot_name)
                        if os.path.exists(source):
                            target = os.path.join(
                                log_plots_dir, 
                                f'{model_type}_{os.path.splitext(plot_name)[0]}_{len(self.loader.train_X)}_{self.session_id}.png'
                            )
                            shutil.copy2(source, target)
                except Exception as e:
                    print_and_log(f"Could not save plots to log folder: {str(e)}", "YELLOW")
                    
        except Exception as e:
            print_and_log(f"Error generating evaluation plots for {model_type}: {str(e)}", "RED")


    def generate_correlation_matrices(self, model_type):
        """Helper method to generate correlation matrices."""
        from src.functions.data_prep.misc_functions import correlation_matrix
        from src.functions.data_prep.raw_features_to_list import raw_features_to_list

        # Regular correlation matrix
        correlation_matrix(
            X=self.loader.train_X[globals()['self.modeller_' + model_type].final_features],
            y=None,
            flag_matrix='all',
            input_data_project_folder=None,
            session_id_folder=self.session_id_folder,
            model_corr=model_type,
            flag_raw='',
            keep_cols=None
        )

        # Raw features correlation matrix
        globals()['self.modeller_' + model_type].raw_features = raw_features_to_list(
            globals()['self.modeller_' + model_type].final_features
        )
        
        try:
            correlation_matrix(
                X=self.loader.train_X[globals()['self.modeller_' + model_type].raw_features],
                y=None,
                flag_matrix='all',
                input_data_project_folder=None,
                session_id_folder=self.session_id_folder,
                model_corr=model_type,
                flag_raw='yes',
                keep_cols=None
            )
        except Exception as e:
            print_and_log(f"[ TRAINING ] Raw features correlation matrix error for model: {model_type}. Error: {e}", "RED")

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

        # x_train = self.loader.train_X_us[globals()['self.modeller_' + model_type].final_features]
        # y_train = self.loader.y_train_us
        #
        # x_test = self.loader.test_X_us[globals()['self.modeller_' + model_type].final_features]
        # y_test = self.loader.y_test_us


        if self.under_sampling:
            globals()['self.modeller_' + model_type].model_fit(
                self.loader.train_X_us[globals()['self.modeller_' + model_type].final_features],
                self.loader.y_train_us,
                self.loader.test_X_us[globals()['self.modeller_' + model_type].final_features], self.loader.y_test_us)
        else:
            globals()['self.modeller_' + model_type].model_fit(
                self.loader.train_X[globals()['self.modeller_' + model_type].final_features],
                self.loader.y_train,
                self.loader.test_X[globals()['self.modeller_' + model_type].final_features], self.loader.y_test)


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
