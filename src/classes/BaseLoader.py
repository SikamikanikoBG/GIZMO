import logging
import pickle

import pandas as pd
from pyarrow import parquet as pq

from src.functions.data_load_functions import check_separator_csv_file, print_and_log, check_files_for_csv_suffix
from src.functions.data_prep.misc_functions import check_if_multiclass_criterion_is_passed, remove_periods_from_df
from src.functions.data_prep.under_sampling import under_sampling_df_based_on_params


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

        if params['under_sampling']:
            self.input_df_full = pq.read_table(
                output_data_folder_name + input_data_project_folder + '/' 'output_data_file_full.parquet')
            self.input_df_full = self.input_df_full.to_pandas()
            self.input_df_full = self.input_df_full.loc[:, ~self.input_df_full.columns.duplicated()]

        with (open(output_data_folder_name + input_data_project_folder + '/' + 'final_features.pkl', "rb")) as openfile:
            self.final_features = pickle.load(openfile)
        print_and_log('\t Data Loaded', 'GREEN')

    def load_from_csv(self, input_data_folder_name, input_data_project_folder):
        print('\n Starting data load... \n')
        logging.info('\n Data load...')
        input_file = check_files_for_csv_suffix(input_data_folder_name, input_data_project_folder)
        # Load csv file
        self.input_df = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file)
        # todo: remove after
        self.input_df = self.input_df.sample(n=10000)
        print(f"Loading file {input_file}")
        logging.info(f"\n Loading file {input_file}")
        return input_file