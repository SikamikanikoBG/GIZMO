import pickle
from importlib import import_module

import pandas as pd
from pyarrow import parquet as pq

from src.functions.data_load_functions import check_separator_csv_file, print_and_log
from src.functions.data_prep.misc_functions import check_if_multiclass_criterion_is_passed, remove_periods_from_df
from src.functions.data_prep.under_sampling import under_sampling_df_based_on_params


def load_from_csv(input_data_folder_name, input_data_project_folder, file):
    df = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + file)
    return df


def load_from_parquet(input_data_folder_name, input_data_project_folder, file):
    df = pd.read_parquet(input_data_folder_name + input_data_project_folder + '/' + file,
                         engine='pyarrow')
    return df


class BaseLoader:
    def __init__(self, params):
        self.additional_files_df_dict = []
        self.in_df, self.in_df_f = pd.DataFrame(), pd.DataFrame()
        self.params = params
        self.main_table = self.params["main_table"]
        self.final_features = None
        self.train_X, self.train_X_us, self.test_X, self.test_X_us, self.t1df, self.t2df, self.t3df = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.y_train, self.y_train_us, self.y_test, self.y_test_us, self.t1df_y, self.t2df_y, self.t3df_y = \
            None, None, None, None, None, None, None

    def data_load_prep(self, in_data_folder, in_data_proj_folder):
        """
        Loads the file in the provided folder. No data manipulations. Checks if criterion is binary and prepares
        dataframe with under-sampling strategy if needed.
        Args:
            in_data_folder: as defined for the session
            in_data_proj_folder: as defined for the session
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
            self.in_df = module_lib.calculate_criterion(self.in_df)

        self.in_df = remove_periods_from_df(self.in_df, self.params)

        if self.params["additional_tables"]:
            for file in self.params["additional_tables"]:
                print_and_log(f"[ ADDITIONAL TABLES ] Loading {file}", "")

                if '.csv' in file:
                    additional_file_df = pd.read_csv(in_data_folder + in_data_proj_folder + "/" + file)
                else:
                    additional_file_df = load_from_parquet(in_data_folder, in_data_proj_folder, file)

                #todo remove comment
                """try:
                    if self.params["custom_calculations"]:
                        additional_file_df = module_lib.run(additional_file_df)
                except Exception as e:
                    print_and_log(f"[ ADDITIONAL TABLES ] Cannot process {file} with custom calcs. Skipping...", "RED")
                    pass"""
                self.additional_files_df_dict.append(additional_file_df)

        if self.params["under_sampling"]:
            self.in_df, self.in_df_f = under_sampling_df_based_on_params(self.in_df, self.params)
            # todo: remove after
            self.in_df[self.params['observation_date_column']] = self.in_df[self.params['observation_date_column']].astype('O')
            self.in_df_f[self.params['observation_date_column']] = self.in_df_f[self.params['observation_date_column']].astype('O')

        check_if_multiclass_criterion_is_passed(self.in_df, self.params)

    def data_load_train(self, output_data_folder_name, input_data_project_folder):
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
