"""
This is a shorter flow in order to speed up the developments.
"""
from pickle import dump

import pandas as pd
from datetime import datetime

import src.functions.data_prep.dates_manipulation as date_funcs
from src.classes.DfAggregator import DfAggregator
from src.classes.OptimalBinning import OptimaBinning
from src.classes.SessionManager import SessionManager
from src.functions.data_prep.misc_functions import split_columns_by_types, switch_numerical_to_object_column, \
    convert_obj_to_cat_and_get_dummies, remove_column_if_not_in_final_features, \
    create_dict_based_on_col_name_contains, create_ratios, correlation_matrix, \
    remove_categorical_cols_with_too_many_values, treating_outliers
from src.functions.data_prep.missing_treatment import missing_values
from src.functions.printing_and_logging import print_end, print_and_log, print_load


class ModuleClass(SessionManager):
    def __init__(self, args):
        SessionManager.__init__(self, args)
        self.predict_session_flag = None

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        self.check1_time = datetime.now()
        self.prepare()
        print_load()
        print_and_log(f'Starting the session for: {self.input_data_project_folder}', 'GREEN')

        self.merging_additional_files_procedure()
        self.check2_time = datetime.now()

        self.data_cleaning()
        self.check3_time = datetime.now()

        # Saving processed data
        self.loader.in_df.to_parquet(
            self.output_data_folder_name + self.input_data_project_folder + '/' + 'output_data_file.parquet')
        if self.under_sampling:
            self.loader.in_df_f.to_parquet(
                self.output_data_folder_name + self.input_data_project_folder + '/' + 'output_data_file_full.parquet')
        with open(self.output_data_folder_name + self.input_data_project_folder + '/' + 'final_features.pkl',
                  'wb') as f:
            dump(self.loader.final_features, f)
        self.check4_time = datetime.now()
        print_end()

    def data_cleaning(self):  # start data cleaning


        print_and_log('[ DATA CLEANING ] Splitting columns by types ', '')
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.in_df,
                                                                                           params=self.params)

        """print_and_log('[ DATA CLEANING ] Treating outliers', '')
        if self.under_sampling:
            self.loader.in_df[numerical_cols], self.loader.in_df_f[numerical_cols] = treating_outliers(
                input_df=self.loader.in_df[numerical_cols],
                secondary_input_df=self.loader.in_df_f[numerical_cols])
        else:
            self.loader.in_df[numerical_cols], _ = treating_outliers(
                input_df=self.loader.in_df[numerical_cols],
                secondary_input_df=pd.DataFrame())"""

        print_and_log('[ DATA CLEANING ] Converting objects to dates', '')
        self.loader.in_df = date_funcs.convert_obj_to_date(self.loader.in_df, object_cols, "_date")
        if self.under_sampling: self.loader.in_df_f = date_funcs.convert_obj_to_date(self.loader.in_df_f, object_cols,
                                                                                     "_date")
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.in_df,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Calculating date differences between date columns', '')
        self.loader.in_df = date_funcs.calculate_date_diff_between_date_columns(self.loader.in_df, dates_cols)
        if self.under_sampling: self.loader.in_df_f = date_funcs.calculate_date_diff_between_date_columns(
            self.loader.in_df_f,
            dates_cols)
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.in_df,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Extracting date characteristics as features', '')
        self.loader.in_df = date_funcs.extract_date_characteristics_from_date_column(self.loader.in_df,
                                                                                     dates_cols)
        if self.under_sampling: self.loader.in_df_f = date_funcs.extract_date_characteristics_from_date_column(
            self.loader.in_df_f,
            dates_cols)
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.in_df,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ]  remove categorical cols with too many values', '')
        object_cols = remove_categorical_cols_with_too_many_values(self.loader.in_df, object_cols)


        # treat specified numerical as objects
        print_and_log('[ DATA CLEANING ]  Switching type from number to object since nb of categories is below 20', "")
        numerical_cols, object_cols = switch_numerical_to_object_column(self.loader.in_df, numerical_cols,
                                                                        object_cols)


        # convert objects to categories and get dummies
        print_and_log('[ DATA CLEANING ]  Converting objects to dummies', "")
        self.loader.in_df, self.loader.in_df_f = convert_obj_to_cat_and_get_dummies(self.loader.in_df,
                                                                                    self.loader.in_df_f,
                                                                                    object_cols, self.params)


        # create cat and dummies dictionaries
        object_cols_cat = create_dict_based_on_col_name_contains(self.loader.in_df.columns.to_list(), '_cat')
        object_cols_dummies = create_dict_based_on_col_name_contains(self.loader.in_df.columns.to_list(), '_dummie')

        # final features to be processed further
        self.loader.final_features = object_cols_dummies + object_cols_cat + numerical_cols

        # Check correlation. Remove low correlation columns from numerical. At this point this is needed to lower the nb of ratios to be created
        print_and_log('[ DATA CLEANING ]  Removing low correlation columns from numerical', '')
        self.remove_final_features_with_low_correlation()
        self.loader.final_features, numerical_cols = remove_column_if_not_in_final_features(self.loader.final_features,
                                                                                            numerical_cols, self.params["columns_to_include"])

        # Feature engineering
        print_and_log('[ DATA CLEANING ] Creating ratios with numerical columns', '')
        self.loader.in_df = create_ratios(df=self.loader.in_df, columns=numerical_cols, columns_to_include=self.params["columns_to_include"])
        if self.under_sampling:
            self.loader.in_df_f = create_ratios(df=self.loader.in_df_f, columns=numerical_cols, columns_to_include=self.params["columns_to_include"])

        ratios_cols = create_dict_based_on_col_name_contains(self.loader.in_df.columns.to_list(), '_ratio_')
        self.loader.final_features = object_cols_dummies + object_cols_cat + numerical_cols + ratios_cols

        # Check correlation
        print_and_log('[ DATA CLEANING ] Removing low correlation columns from ratios', '')
        self.remove_final_features_with_low_correlation()
        self.loader.final_features, numerical_cols = remove_column_if_not_in_final_features(self.loader.final_features,
                                                                                            ratios_cols, self.params["columns_to_include"])
        print_and_log(f'[ DATA CLEANING ] Final features so far {len(self.loader.final_features)}', '')

        if self.optimal_binning_columns:
            self.optimal_binning_procedure()

        # Finalizing the dataframes
        self.loader.in_df = missing_values(df=self.loader.in_df, missing_treatment=self.missing_treatment,
                                           input_data_project_folder=self.input_data_project_folder)
        if self.under_sampling:
            self.loader.in_df_f = missing_values(df=self.loader.in_df_f,
                                                 missing_treatment=self.missing_treatment,
                                                 input_data_project_folder=self.input_data_project_folder)

        for col in self.params["columns_to_include"]:
            if col not in self.loader.final_features[:]:
                self.loader.final_features.append(col)

        # check if some final features were deleted
        for el in self.loader.final_features[:]:
            if el not in self.loader.in_df.columns:
                self.loader.final_features.remove(el)

        print_and_log(f'[ DATA CLEANING ] Final features so far {len(self.loader.final_features)}', '')

        all_columns = self.loader.in_df.columns.to_list()

        if self.under_sampling:
            for el in all_columns[:]:
                if el not in self.loader.in_df_f.columns:
                    all_columns.remove(el)
                    if el in self.loader.final_features[:]:
                        self.loader.final_features.remove(el)

        # self.loader.in_df = self.loader.in_df[all_columns].copy()
        if self.under_sampling:
            self.loader.in_df_f = self.loader.in_df_f[all_columns].copy()

        # remove duplicated features
        results_check = []
        for el in self.loader.final_features:
            if el not in results_check:
                results_check.append(el)
        self.loader.final_features = results_check.copy()

    def remove_final_features_with_low_correlation(self):
        self.loader.final_features = correlation_matrix(X=self.loader.in_df[self.loader.final_features],
                                                        y=self.loader.in_df[self.criterion_column],
                                                        input_data_project_folder=self.input_data_project_folder,
                                                        flag_matrix=None,
                                                        session_id_folder=None, model_corr='', flag_raw='',
                                                        keep_cols=self.params["columns_to_include"])

    def optimal_binning_procedure(self):
        binned_numerical = self.optimal_binning_columns
        # Optimal binning
        optimal_binning_obj = OptimaBinning(df=self.loader.in_df, df_full=self.loader.in_df_f,
                                            columns=binned_numerical,
                                            criterion_column=self.criterion_column,
                                            final_features=self.loader.final_features,
                                            observation_date_column=self.observation_date_column,
                                            params=self.params)
        print_and_log(' Starting numerical features Optimal binning (max 4 bins) based on train df_to_aggregate ', '')
        self.loader.in_df, self.loader.in_df_f, binned_numerical = optimal_binning_obj.run_optimal_binning_multiprocess()
        self.loader.final_features = self.loader.final_features + binned_numerical
        self.loader.final_features = list(set(self.loader.final_features))
        self.loader.final_features, _ = remove_column_if_not_in_final_features(self.loader.final_features,
                                                                               self.loader.final_features[:], self.params["columns_to_include"])

        # Check correlation
        print_and_log(' Checking correlation one more time now with binned ', '')
        self.remove_final_features_with_low_correlation()
        self.loader.final_features, binned_numerical = remove_column_if_not_in_final_features(
            self.loader.final_features,
            binned_numerical, self.params["columns_to_include"])

    def merging_additional_files_procedure(self):
        count = 0

        if self.params["additional_tables"]:
            for file in self.params["additional_tables"]:
                print_and_log(f"[ ADDITIONAL SOURCES ] Merging {file}", "GREEN")

                merge_group_cols = self.params["additional_tables"][file]
                merge_group_cols_input_df = merge_group_cols.copy()
                merge_group_cols_input_df.append(self.params["criterion_column"])
                aggregator = DfAggregator(params=self.params)
                merged_temp = aggregator.aggregation_procedure(
                    df_to_aggregate=self.loader.additional_files_df_dict[count],
                    columns_to_group=merge_group_cols)

                suffix = "_" + file.split('.')[0]
                for el in merged_temp.columns:
                    if el not in merge_group_cols_input_df:
                        merged_temp[el + suffix] = merged_temp[el].copy()
                        del merged_temp[el]

                if len(merged_temp) > 1:
                    periods = ['T']

                    # if 1!=1:
                    if self.observation_date_column in merge_group_cols:
                        for period in periods:
                            merged_temp[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(merged_temp[self.observation_date_column]).dt.to_period(period)
                            self.loader.in_df[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(self.loader.in_df[self.observation_date_column]).dt.to_period(period)
                            if self.under_sampling: self.loader.in_df_f[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(self.loader.in_df_f[self.observation_date_column]).dt.to_period(period)

                            merge_group_cols_periods = merge_group_cols.copy()
                            merge_group_cols_periods.remove(self.observation_date_column)
                            merge_group_cols_periods.append(self.observation_date_column + '_temp')

                            self.loader.in_df = self.loader.in_df.merge(merged_temp, how='left',
                                                                        on=merge_group_cols,
                                                                        suffixes=("", f"_{period}{suffix}"))
                            missing_cols = self.loader.in_df[
                                self.loader.in_df.columns[self.loader.in_df.isnull().mean() > 0.80]].columns.to_list()
                            self.loader.in_df = self.loader.in_df.drop(columns=missing_cols)

                            if self.under_sampling:
                                self.loader.in_df_f = self.loader.in_df_f.merge(merged_temp,
                                                                                                    how='left',
                                                                                                    on=merge_group_cols,
                                                                                                    suffixes=("",
                                                                                                              f"_{period}{suffix}"))
                    else:
                        self.loader.in_df = self.loader.in_df.merge(merged_temp, how='left', on=merge_group_cols,
                                                                    suffixes=("", f"{suffix}"))
                        if self.under_sampling: self.loader.in_df_f = self.loader.in_df_f.merge(merged_temp, how='left',
                                                                                                on=merge_group_cols,
                                                                                                suffixes=(
                                                                                                    "", f"{suffix}"))
                count += 1
