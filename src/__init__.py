import json
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
import xgboost
from colorama import Fore
from colorama import Style
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from src.functions.data_load_functions import check_files_for_csv_suffix, check_separator_csv_file
from src.functions.data_prep_functions import check_if_multiclass_criterion_is_passed, remove_periods_from_df, \
    under_sampling_df_based_on_params
from src.functions.modelling_functions import cut_into_bands, get_metrics
from src.functions.printing_and_logging_functions import print_and_log


class SessionManager:
    def __init__(self, args):
        self.start = datetime.now()  # .strftime("%Y-%m-%d_%H:%M:%S")
        self.end_time = None
        self.run_time = None
        self.session_id = None
        self.params = None
        self.args = args
        self.session_to_eval = self.args.session
        self.model_arg = self.args.model
        # self.run_info = self.args.run
        self.project_name = self.args.project
        if not self.args.tag:
            self.tag = 'no_tag'
        else:
            self.tag = self.args.tag

        # folder structure properties
        if self.args.data_prep_module:
            self.log_file_name = 'load_' + self.args.data_prep_module + '_' + self.project_name + '_' + str(
                self.start) + '_' + self.tag + ".log"
        elif self.args.train_module:
            self.log_file_name = 'train_' + self.args.train_module + '_' + self.project_name + '_' + str(
                self.start) + '_' + self.tag + ".log"
        self.log_folder_name = './logs/'
        self.session_folder_name = './sessions/'
        self.session_id_folder = None
        self.input_data_folder_name = './input_data/'
        self.input_data_project_folder = self.project_name
        self.output_data_folder_name = './output_data/'
        self.functions_folder_name = './src/'
        self.params_folder_name = './params/'

        self.start_logging()

        # Import parameters
        try:
            with open('params/params_' + self.project_name + '.json') as json_file:
                self.params = json.load(json_file)
            self.criterion_column = self.params['criterion_column']
            self.missing_treatment = self.params["missing_treatment"]
            self.observation_date_column = self.params["observation_date_column"]
            self.columns_to_exclude = self.params["columns_to_exclude"]
            self.periods_to_exclude = self.params["periods_to_exclude"]
            self.t1df_period = self.params["t1df"]
            self.t2df_period = self.params["t2df"]
            self.t3df_period = self.params["t3df"]
            self.lr_features = self.params["lr_features"]
            self.cut_offs = self.params["cut_offs"]
            self.under_sampling = self.params['under_sampling']
        except Exception as e:
            print(Fore.RED + 'ERROR: params file not available' + Style.RESET_ALL)
            print(e)
            logging.error(e)
            quit()

        self.loader = BaseLoader()
        if self.args.data_prep_module:
            self.loader.data_load_prep(input_data_folder_name=self.input_data_folder_name,
                                       input_data_project_folder=self.input_data_project_folder,
                                       params=self.params)
        elif self.args.train_module:
            self.loader.data_load_train(output_data_folder_name=self.output_data_folder_name,
                                        input_data_project_folder=self.input_data_project_folder,
                                        params=self.params)

    def prepare(self):
        """
        Orchestrates the preparation of the session run
        Returns:

        """
        self.create_folders()

    def create_folders(self):
        """
        Creates the folder structure if some of it is missing
        Returns:

        """
        if not os.path.isdir(self.log_folder_name):
            os.mkdir(self.log_folder_name)
        if not os.path.isdir(self.session_folder_name):
            os.mkdir(self.session_folder_name)
        if not os.path.isdir(self.input_data_folder_name):
            os.mkdir(self.input_data_folder_name)
        if not os.path.isdir(self.output_data_folder_name):
            os.mkdir(self.output_data_folder_name)
        if not os.path.isdir(self.functions_folder_name):
            os.mkdir(self.functions_folder_name)
        if not os.path.isdir(self.params_folder_name):
            os.mkdir(self.params_folder_name)

        # Create output data project folder
        if not os.path.isdir(self.output_data_folder_name + self.input_data_project_folder + '/'):
            os.mkdir(self.output_data_folder_name + self.input_data_project_folder + '/')

    def start_logging(self):
        """
        Starts the logging process for the session and creates the log file
        Returns:

        """
        # logging
        if not os.path.isfile(self.log_folder_name + self.log_file_name):
            open(self.log_folder_name + self.log_file_name, 'w').close()

        logging.basicConfig(
            filename=self.log_folder_name + self.log_file_name,
            level=logging.INFO, format='%(asctime)s - %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        print_and_log('Start logging', 'YELLOW')

    def run_time_calc(self):
        """
        Calculates the time delta that the session took to run
        Returns:

        """
        self.end_time = datetime.now()
        self.run_time = round(float((self.end_time - self.start).total_seconds()), 2)


class BaseLoader:
    def __init__(self):
        self.input_df, self.input_df_full = pd.DataFrame(), pd.DataFrame()
        self.final_features = None
        self.train_X, self.train_X_us, self.test_X, self.test_X_us, self.t1df, self.t2df, self.t3df = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.y_train, self.y_train_us, self.y_test, self.y_test_us, self.t1df_y, self.t2df_y, self.t3df_y = \
            None, None, None, None, None, None, None

    def data_load_prep(self, input_data_folder_name, input_data_project_folder, params):
        """
        Loads the file in the provided folder. No data manipulations. Checks if criterion is binary and prepares
        dataframe with under-sampling strategy if needed.
        Args:
            input_data_folder_name: as defined for the session
            input_data_project_folder: as defined for the session
            params: from params file for the project
        """
        # todo: add if logic - when loading from csv and when loading from API
        input_file = self.load_from_csv(input_data_folder_name, input_data_project_folder)
        self.input_df = check_separator_csv_file(input_data_folder_name, input_data_project_folder, self.input_df,
                                                 input_file)
        check_if_multiclass_criterion_is_passed(self.input_df, params)
        self.input_df = remove_periods_from_df(self.input_df, params)

        if params["under_sampling"]:
            self.input_df, self.input_df_full = under_sampling_df_based_on_params(self.input_df, params)

    def data_load_train(self, output_data_folder_name, input_data_project_folder, params):
        print_and_log('\n Loading data \n', 'YELLOW')
        self.input_df = pq.read_table(
            output_data_folder_name + input_data_project_folder + '/' 'output_data_file.parquet')
        self.input_df = self.input_df.to_pandas()
        # checking for duplicated columns and treating them
        self.input_df = self.input_df.loc[:, ~self.input_df.columns.duplicated()]

        # todo remove print
        print(self.input_df.columns.tolist())

        if params['under_sampling']:
            self.input_df_full = pq.read_table(
                output_data_folder_name + input_data_project_folder + '/' 'output_data_file_full.parquet')
            self.input_df_full = self.input_df_full.to_pandas()
            self.input_df_full = self.input_df_full.loc[:, ~self.input_df_full.columns.duplicated()]

            # todo remove print
            print(self.input_df.columns.tolist())

        with (open(output_data_folder_name + input_data_project_folder + '/' + 'final_features.pkl', "rb")) as openfile:
            self.final_features = pickle.load(openfile)
            # todo remove print
            print(self.final_features)
        print_and_log('\t Data Loaded', 'GREEN')

    def load_from_csv(self, input_data_folder_name, input_data_project_folder):
        print('\n Starting data load... \n')
        logging.info('\n Data load...')
        input_file = check_files_for_csv_suffix(input_data_folder_name, input_data_project_folder)
        # Load csv file
        self.input_df = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file)
        # todo: removafter
        self.input_df = self.input_df.sample(n=10000)
        print(f"Loading file {input_file}")
        logging.info(f"\n Loading file {input_file}")
        return input_file


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
