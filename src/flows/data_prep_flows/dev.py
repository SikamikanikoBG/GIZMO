"""
This is a shorter flow in order to speed up the developments.
"""
from pickle import dump

import pandas as pd

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

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        self.prepare()
        print_load()
        print_and_log(f'Starting the session for: {self.input_data_project_folder}', 'GREEN')

        self.merging_additional_files_procedure()

        self.data_cleaning()

        # todo: remove after
        cols = []
        for col in self.loader.input_df_full.columns:
            if 'file' in col:
                cols.append(col)
        cols.append('MSENDOS')
        self.loader.input_df_full[self.loader.input_df_full['MSENDOS'].isin([42040052111100, 42081073691100])][
            cols].to_csv(
            "temp.csv", index=False)

        # Saving processed data
        self.loader.input_df.to_parquet(
            self.output_data_folder_name + self.input_data_project_folder + '/' + 'output_data_file.parquet')
        if self.under_sampling:
            self.loader.input_df_full.to_parquet(
                self.output_data_folder_name + self.input_data_project_folder + '/' + 'output_data_file_full.parquet')
        with open(self.output_data_folder_name + self.input_data_project_folder + '/' + 'final_features.pkl',
                  'wb') as f:
            dump(self.loader.final_features, f)
        print_end()

    def data_cleaning(self):  # start data cleaning

        print_and_log('[ DATA CLEANING ] Splitting columns by types ', '')
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.input_df,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Treating outliers', '')
        self.loader.input_df[numerical_cols], self.loader.input_df_full[numerical_cols] = treating_outliers(
            input_df=self.loader.input_df[numerical_cols],
            secondary_input_df=self.loader.input_df_full[numerical_cols])

        print_and_log('[ DATA CLEANING ] Converting objects to dates', '')
        self.loader.input_df = date_funcs.convert_obj_to_date(self.loader.input_df, object_cols, "_date")
        self.loader.input_df_full = date_funcs.convert_obj_to_date(self.loader.input_df_full, object_cols, "_date")
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.input_df,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Calculating date differences between date columns', '')
        self.loader.input_df = date_funcs.calculate_date_diff_between_date_columns(self.loader.input_df, dates_cols)
        self.loader.input_df_full = date_funcs.calculate_date_diff_between_date_columns(self.loader.input_df_full,
                                                                                        dates_cols)
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.input_df,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Extracting date characteristics as features', '')
        self.loader.input_df = date_funcs.extract_date_characteristics_from_date_column(self.loader.input_df,
                                                                                        dates_cols)
        self.loader.input_df_full = date_funcs.extract_date_characteristics_from_date_column(self.loader.input_df_full,
                                                                                             dates_cols)
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.loader.input_df,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ]  remove categorical cols with too many values', '')
        object_cols = remove_categorical_cols_with_too_many_values(self.loader.input_df, object_cols)

        # treat specified numerical as objects
        print_and_log('[ DATA CLEANING ]  Switching type from number to object since nb of categories is below 20', "")
        numerical_cols, object_cols = switch_numerical_to_object_column(self.loader.input_df, numerical_cols,
                                                                        object_cols)

        # convert objects to categories and get dummies
        print_and_log('[ DATA CLEANING ]  Converting objects to dummies', "")
        self.loader.input_df, self.loader.input_df_full = convert_obj_to_cat_and_get_dummies(self.loader.input_df,
                                                                                             self.loader.input_df_full,
                                                                                             object_cols, self.params)

        # create cat and dummies dictionaries
        object_cols_cat = create_dict_based_on_col_name_contains(self.loader.input_df.columns.to_list(), '_cat')
        object_cols_dummies = create_dict_based_on_col_name_contains(self.loader.input_df.columns.to_list(), '_dummie')

        # final features to be processed further
        self.loader.final_features = object_cols_dummies + object_cols_cat + numerical_cols

        # Check correlation. Remove low correlation columns from numerical. At this point this is needed to lower the nb of ratios to be created
        print_and_log('[ DATA CLEANING ]  Removing low correlation columns from numerical', '')
        self.remove_final_features_with_low_correlation()
        self.loader.final_features, numerical_cols = remove_column_if_not_in_final_features(self.loader.final_features,
                                                                                            numerical_cols)

        # Feature engineering
        print_and_log('[ DATA CLEANING ] Creating ratios with numerical columns', '')
        self.loader.input_df = create_ratios(df=self.loader.input_df, columns=numerical_cols)
        if self.under_sampling:
            self.loader.input_df_full = create_ratios(df=self.loader.input_df_full, columns=numerical_cols)

        ratios_cols = create_dict_based_on_col_name_contains(self.loader.input_df.columns.to_list(), '_ratio_')
        self.loader.final_features = object_cols_dummies + object_cols_cat + numerical_cols + ratios_cols

        # Check correlation
        print_and_log('[ DATA CLEANING ] Removing low correlation columns from ratios', '')
        self.remove_final_features_with_low_correlation()
        self.loader.final_features, numerical_cols = remove_column_if_not_in_final_features(self.loader.final_features,
                                                                                            ratios_cols)

        if self.optimal_binning_columns:
            self.optimal_binning_procedure()

        # Finalizing the dataframes
        self.loader.input_df = missing_values(df=self.loader.input_df, missing_treatment=self.missing_treatment,
                                              input_data_project_folder=self.input_data_project_folder)
        if self.under_sampling:
            self.loader.input_df_full = missing_values(df=self.loader.input_df_full,
                                                       missing_treatment=self.missing_treatment,
                                                       input_data_project_folder=self.input_data_project_folder)

        # check if some final features were deleted
        for el in self.loader.final_features[:]:
            if el not in self.loader.input_df.columns:
                self.loader.final_features.remove(el)

        all_columns = self.loader.input_df.columns.to_list()

        for el in all_columns[:]:
            if el not in self.loader.input_df_full.columns:
                all_columns.remove(el)
                if el in self.loader.final_features[:]:
                    self.loader.final_features.remove(el)

        # self.loader.input_df = self.loader.input_df[all_columns].copy()
        if self.under_sampling:
            self.loader.input_df_full = self.loader.input_df_full[all_columns].copy()

    def remove_final_features_with_low_correlation(self):
        self.loader.final_features = correlation_matrix(X=self.loader.input_df[self.loader.final_features],
                                                        y=self.loader.input_df[self.criterion_column],
                                                        input_data_project_folder=self.input_data_project_folder,
                                                        flag_matrix=None,
                                                        session_id_folder=None, model_corr='', flag_raw='')

    def optimal_binning_procedure(self):
        binned_numerical = self.optimal_binning_columns
        # Optimal binning
        optimal_binning_obj = OptimaBinning(df=self.loader.input_df, df_full=self.loader.input_df_full,
                                            columns=binned_numerical,
                                            criterion_column=self.criterion_column,
                                            final_features=self.loader.final_features,
                                            observation_date_column=self.observation_date_column,
                                            params=self.params)
        print_and_log(' Starting numerical features Optimal binning (max 4 bins) based on train df_to_aggregate ', '')
        self.loader.input_df, self.loader.input_df_full, binned_numerical = optimal_binning_obj.run_optimal_binning_multiprocess()
        self.loader.final_features = self.loader.final_features + binned_numerical
        self.loader.final_features = list(set(self.loader.final_features))
        self.loader.final_features, _ = remove_column_if_not_in_final_features(self.loader.final_features,
                                                                               self.loader.final_features[:])

        # Check correlation
        print_and_log(' Checking correlation one more time now with binned ', '')
        self.remove_final_features_with_low_correlation()
        self.loader.final_features, binned_numerical = remove_column_if_not_in_final_features(
            self.loader.final_features,
            binned_numerical)

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
                    periods = ['M', 'D', 'H', 'T']

                    # if 1!=1:
                    if self.observation_date_column in merge_group_cols:
                        for period in periods:
                            merged_temp[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(merged_temp[self.observation_date_column]).dt.to_period(period)
                            self.loader.input_df[self.observation_date_column + '_temp'] =  \
                                pd.to_datetime(self.loader.input_df[self.observation_date_column]).dt.to_period(period)
                            self.loader.input_df_full[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(self.loader.input_df_full[self.observation_date_column]).dt.to_period(period)

                            merge_group_cols_periods = merge_group_cols.copy()
                            merge_group_cols_periods.remove(self.observation_date_column)
                            merge_group_cols_periods.append(self.observation_date_column + '_temp')

                            self.loader.input_df = self.loader.input_df.merge(merged_temp, how='left',
                                                                              on=merge_group_cols,
                                                                              suffixes=("", f"_{period}{suffix}"))
                            missing_cols = self.loader.input_df[self.loader.input_df.columns[self.loader.input_df.isnull().mean() > 0.80]].columns.to_list()
                            self.loader.input_df = self.loader.input_df.drop(columns=missing_cols)

                            if self.under_sampling:
                                self.loader.input_df_full = self.loader.input_df_full.merge(merged_temp, how='left',
                                                                                            on=merge_group_cols,
                                                                                            suffixes=(
                                                                                            "", f"_{period}{suffix}"))
                    else:
                        self.loader.input_df = self.loader.input_df.merge(merged_temp, how='left',
                                                                          on=merge_group_cols,
                                                                          suffixes=("", f"{suffix}"))
                        if self.under_sampling:
                            self.loader.input_df_full = self.loader.input_df_full.merge(merged_temp, how='left',
                                                                                        on=merge_group_cols,
                                                                                        suffixes=("", f"{suffix}"))
                count += 1
