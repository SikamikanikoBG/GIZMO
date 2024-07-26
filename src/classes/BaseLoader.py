import pickle
from importlib import import_module
import os

import pandas as pd
from pyarrow import parquet as pq

from src.functions.data_load_functions import check_separator_csv_file, print_and_log
from src.functions.data_prep.misc_functions import remove_periods_from_df
from src.functions.data_prep.under_sampling import under_sampling_df_based_on_params


def load_from_csv(input_data_folder_name: str, input_data_project_folder: str, file: str):
    """
    Load a CSV file from the specified input data folder and project folder.

    Args:
        input_data_folder_name (str): The name of the input data folder.

        input_data_project_folder (str): The name of the input data project folder.

        file (str): The name of the CSV file to load.

    Returns:
        pandas.DataFrame: The loaded CSV file as a DataFrame.
    """
    df = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + file)
    return df


def load_from_parquet(input_data_folder_name: str, input_data_project_folder: str, file: str):
    """
    Load a Parquet file from the specified input data folder and project folder.

    Args:
        input_data_folder_name (str): The name of the input data folder.

        input_data_project_folder (str): The name of the input data project folder.

        file (str): The name of the Parquet file to load.

    Returns:
        pandas.DataFrame: The loaded Parquet file as a DataFrame.
    """
    df = pd.read_parquet(input_data_folder_name + input_data_project_folder + '/' + file,
                         engine='pyarrow')
    return df


class BaseLoader:
    def __init__(self, params, predict_module):
        """
        Initialize the BaseLoader object with parameters and prediction module.

        Args:
            params (dict): Dictionary of parameters.

            predict_module: The prediction module.

        Attributes:
            additional_files_df_dict (list): List to store additional files as DataFrames.

            in_df (pandas.DataFrame): DataFrame for input data.

            in_df_f (pandas.DataFrame): DataFrame for input data with under-sampling.

            params (dict): Dictionary of parameters.

            main_table (str): Name of the main table.

            final_features: Placeholder for final features.

            train_X, train_X_us, test_X, test_X_us, t1df, t2df, t3df (pandas.DataFrame): DataFrames for training and testing.

            y_train, y_train_us, y_test, y_test_us, t1df_y, t2df_y, t3df_y: Placeholders for target variables.

            predict_module: The prediction module.
        """
        self.additional_files_df_dict = []
        self.in_df, self.in_df_f = pd.DataFrame(), pd.DataFrame()
        self.params = params
        self.main_table = self.params["main_table"]
        self.final_features = None
        self.train_X, self.train_X_us, self.test_X, self.test_X_us, self.t1df, self.t2df, self.t3df = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.y_train, self.y_train_us, self.y_test, self.y_test_us, self.t1df_y, self.t2df_y, self.t3df_y = \
            None, None, None, None, None, None, None
        self.predict_module = predict_module
        # self.is_multiclass = True if self.y_train.nunique() > 2 else False  # Flag that determines if we have multiclass or not

    def data_load_prep(self, in_data_folder: str, in_data_proj_folder: str):
        """
        Load and prepare data without manipulation. Check for binary criterion and apply under-sampling if needed.

        Args:
            in_data_folder (str): Input data folder.

            in_data_proj_folder (str): Input data project folder.
        """

        # todo: add if logic - when loading from csv and when loading from API
        print_and_log(f"[ LOADING ] Loading file {self.main_table}", "")
        if '.csv' in self.main_table:
            self.in_df = load_from_csv(in_data_folder, in_data_proj_folder, self.main_table)
            self.in_df = check_separator_csv_file(in_data_folder, in_data_proj_folder, self.in_df,
                                                  self.main_table)
        else:
            self.in_df = load_from_parquet(in_data_folder, in_data_proj_folder, self.main_table)

        if self.params["custom_calculations"]:
            module_lib = import_module(f'src.custom_calculations.{self.params["custom_calculations"]}')
            self.in_df = module_lib.run(self.in_df)
            self.in_df = module_lib.calculate_criterion(self.in_df, self.predict_module)

        self.in_df = remove_periods_from_df(self.in_df, self.params)

        if self.params["additional_tables"]:
            for file in self.params["additional_tables"]:
                print_and_log(f"[ ADDITIONAL TABLES ] Loading {file}", "")

                if '.csv' in file:
                    additional_file_df = pd.read_csv(in_data_folder + in_data_proj_folder + "/" + file)
                else:
                    additional_file_df = load_from_parquet(in_data_folder, in_data_proj_folder, file)

                try:
                    if self.params["custom_calculations"]:
                        additional_file_df = module_lib.run(additional_file_df)
                except Exception as e:
                    print_and_log(f"[ ADDITIONAL TABLES ] Cannot process {file} with custom calcs. Skipping...", "RED")
                    pass
                self.additional_files_df_dict.append(additional_file_df)

        if self.params["under_sampling"]:
            self.in_df, self.in_df_f = under_sampling_df_based_on_params(self.in_df, self.params) # Returns:
                                                                                                  # input_df: DataFrame, under-sampled DataFrame
                                                                                                  # input_df_full: DataFrame, original full DataFrame
            # todo: remove after
            self.in_df[self.params['observation_date_column']] = self.in_df[self.params['observation_date_column']].astype('O')
            self.in_df_f[self.params['observation_date_column']] = self.in_df_f[self.params['observation_date_column']].astype('O')

        # Asserts
        assert os.path.exists(
            in_data_folder + in_data_proj_folder), f"Input data path does not exist: {in_data_folder + in_data_proj_folder}"

        assert not self.in_df.empty, "Loaded DataFrame is empty"

        # After under-sampling (if enabled):
        if self.params["under_sampling"]:
            assert not self.in_df_f.empty, "Under-sampled DataFrame is empty"
            assert self.in_df.shape[0] <= self.in_df_f.shape[
                0], "Under-sampled DataFrame should not be larger than the original DataFrame"

    def data_load_train(self, output_data_folder_name, input_data_project_folder):
        """
        Load training data from the specified output data folder and input data project folder.

        Args:
            output_data_folder_name (str): The name of the output data folder.

            input_data_project_folder (str): The name of the input data project folder.

        Loads the data, checks for duplicated columns, handles under-sampling if specified, and loads final features.
        """
        print_and_log('[ LOADING ] Loading data', 'GREEN')
        self.in_df = pq.read_table(
            output_data_folder_name + input_data_project_folder + '/' 'output_data_file.parquet')
        self.in_df = self.in_df.to_pandas()
        # checking for duplicated columns and treating them
        self.in_df = self.in_df.loc[:, ~self.in_df.columns.duplicated()]

        if self.params['under_sampling']:
            self.in_df_f = pq.read_table(
                output_data_folder_name + input_data_project_folder + '/' 'output_data_file_full.parquet')
            self.in_df_f = self.in_df_f.to_pandas()
            self.in_df_f = self.in_df_f.loc[:, ~self.in_df_f.columns.duplicated()]

        with (open(output_data_folder_name + input_data_project_folder + '/' + 'final_features.pkl', "rb")) as openfile:
            self.final_features = pickle.load(openfile)
        print_and_log('[ LOADING ] Data Loaded', 'GREEN')

        # Asserts
        assert os.path.exists(
            output_data_folder_name + input_data_project_folder), f"Output data path does not exist: {output_data_folder_name + input_data_project_folder}"
        assert os.path.exists(
            output_data_folder_name + input_data_project_folder + '/' + 'output_data_file.parquet'), "Output data file not found"

        assert not self.in_df.empty, "Loaded training DataFrame is empty"

        assert os.path.exists(
            output_data_folder_name + input_data_project_folder + '/' + 'final_features.pkl'), "Final features file not found"

        assert self.final_features is not None, "Final features not loaded correctly"

        if self.params['under_sampling']:
            assert os.path.exists(
                output_data_folder_name + input_data_project_folder + '/' + 'output_data_file_full.parquet'), "Output data file (full) not found. Are you undersampling?"
            assert not self.in_df_f.empty, "Loaded full training DataFrame (for under-sampling) is empty. Are you undersampling?"
