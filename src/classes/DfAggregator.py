import pandas as pd

import src.functions.data_prep.dates_manipulation as date_funcs
from src import print_and_log
from src.functions.data_prep.misc_functions import remove_categorical_cols_with_too_many_values, \
    switch_numerical_to_object_column, create_dict_based_on_col_name_contains
from src.functions.data_prep.misc_functions import split_columns_by_types


class DfAggregator:
    def __init__(self, params, criterion_and_merging_columns_input_df):
        self.df_to_return = pd.DataFrame()
        self.final_features_aggregation = None
        self.columns_to_group = None
        self.df_to_aggregate = None
        self.params = params
        self.criterion_and_merging_columns_input_df = criterion_and_merging_columns_input_df
        self.criterion_column = self.params["criterion_column"]

    def aggregation_procedure(self, df_to_aggregate, columns_to_group):
        self.df_to_aggregate = df_to_aggregate
        self.columns_to_group = columns_to_group

        self.data_cleaning_procedure()
        self.test_first_row()
        self.test_aggregation_by_period()

        self.remove_equal_columns()

        # col type
        # manipulate dates
        # group date
        # for col in df_to_aggregate.columns:

        return self.df_to_return

    def data_cleaning_procedure(self):
        print_and_log('[ DATA CLEANING ] Splitting columns by types \n', '')
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.df_to_aggregate,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Converting objects to dates', '')
        self.df_to_aggregate = date_funcs.convert_obj_to_date(self.df_to_aggregate, object_cols)
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.df_to_aggregate,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Calculating date differences between date columns', '')
        self.df_to_aggregate = date_funcs.calculate_date_diff_between_date_columns(self.df_to_aggregate, dates_cols)
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.df_to_aggregate,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ] Extracting date characteristics as features', '')
        self.df_to_aggregate = date_funcs.extract_date_characteristics_from_date_column(self.df_to_aggregate,
                                                                                        dates_cols)
        categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=self.df_to_aggregate,
                                                                                           params=self.params)

        print_and_log('[ DATA CLEANING ]  remove categorical cols with too many values', '')
        object_cols = remove_categorical_cols_with_too_many_values(self.df_to_aggregate, object_cols)

        # treat specified numerical as objects
        print_and_log('[ DATA CLEANING ]  Switching type from number to object since nb of categories is below 20', "")
        numerical_cols, object_cols = switch_numerical_to_object_column(self.df_to_aggregate, numerical_cols,
                                                                        object_cols)

        # convert objects to categories and get dummies
        # todo fix funct to accept withotu input_full
        print_and_log('[ DATA CLEANING ]  Converting objects to dummies', "")
        # self.df_to_aggregate, _ = convert_obj_to_cat_and_get_dummies(self.df_to_aggregate, pd.DataFrame, object_cols, self.params)

        # create cat and dummies dictionaries
        object_cols_cat = create_dict_based_on_col_name_contains(self.df_to_aggregate.columns.to_list(), '_cat')
        object_cols_dummies = create_dict_based_on_col_name_contains(self.df_to_aggregate.columns.to_list(), '_dummie')

        # final features to be processed further
        self.final_features_aggregation = object_cols_dummies + object_cols_cat + numerical_cols

    def test_first_row(self):
        temp_df = self.df_to_aggregate.copy()

        # row nb
        temp_df['RN'] = temp_df.sort_values(self.columns_to_group, ascending=True).groupby(
            self.columns_to_group).cumcount() + 1

        temp_df = temp_df.loc[temp_df['RN'] == 1].copy()
        selected_columns = []

        for col in self.final_features_aggregation:
            correlation = round(
                self.criterion_and_merging_columns_input_df[self.criterion_column].corr(temp_df[col], method='pearson'),
                2)

            if 0.5 > abs(correlation) > 0.05:
                print_and_log(f"[ ADDITIONAL DATA ] Adding {col} the correlation vs criterion {correlation}.", 'GREEN')
                selected_columns.append(col)

        for col in selected_columns:
            temp_df[col + "_RN1"] = temp_df[col].copy()
            del temp_df[col]
        # temp_df[selected_columns] = temp_df[selected_columns].add_suffix("_RN1")
        self.df_to_return = pd.concat([temp_df, self.df_to_return], ignore_index=True, sort=False)

    def test_aggregation_by_period(self):
        pass

    def remove_equal_columns(self):
        pass
