import pandas as pd

import src.functions.data_prep.dates_manipulation as date_funcs
from src import print_and_log
from src.functions.data_prep.misc_functions import remove_categorical_cols_with_too_many_values, \
    switch_numerical_to_object_column, create_dict_based_on_col_name_contains
from src.functions.data_prep.misc_functions import split_columns_by_types


class DfAggregator:
    def __init__(self, params):
        self.df_to_return = pd.DataFrame()
        self.final_features_aggregation = None
        self.columns_to_group = None
        self.df_to_aggregate = None
        self.params = params
        self.criterion_column = self.params["criterion_column"]

    def aggregation_procedure(self, df_to_aggregate, columns_to_group):
        self.df_to_aggregate = df_to_aggregate
        self.columns_to_group = columns_to_group
        self.df_to_return = self.df_to_aggregate[self.columns_to_group].copy()
        self.df_to_return = self.df_to_return.drop_duplicates(keep='last')

        self.merge_first_and_last_rows()
        self.merge_aggregation_by_period()

        return self.df_to_return

    def merge_first_and_last_rows(self):
        temp_df = self.df_to_aggregate.copy()

        # row nb
        temp_df_agg_sum = temp_df.copy().groupby(self.columns_to_group).sum()
        temp_df_agg_count = temp_df.copy().groupby(self.columns_to_group).count()
        temp_df['RN1'] = temp_df.sort_values(self.columns_to_group, ascending=True).groupby(
            self.columns_to_group).cumcount() + 1
        temp_df['RN_1'] = temp_df.sort_values(self.columns_to_group, ascending=False).groupby(
            self.columns_to_group).cumcount() + 1

        temp_df_rn1 = temp_df.loc[temp_df['RN1'] == 1].copy()
        temp_df_rn_1 = temp_df.loc[temp_df['RN_1'] == 1].copy()

        del temp_df
        temp_df_rn1 = temp_df_rn1.drop(['RN1', 'RN_1'], axis=1)
        temp_df_rn_1 = temp_df_rn_1.drop(['RN1', 'RN_1'], axis=1)

        for col in temp_df_rn1.columns:
            if col in self.columns_to_group:
                pass
            else:
                temp_df_rn1[col + "_RN1"] = temp_df_rn1[col].copy()
                temp_df_rn_1[col + "_RN_1"] = temp_df_rn_1[col].copy()
                del temp_df_rn1[col]
                del temp_df_rn_1[col]

        for col in temp_df_agg_sum.columns:
            if col in self.columns_to_group:
                pass
            else:
                temp_df_agg_sum[col + "_agg_sum"] = temp_df_agg_sum[col].copy()
                temp_df_agg_count[col + "_agg_count"] = temp_df_agg_count[col].copy()
                del temp_df_agg_sum[col]
                del temp_df_agg_count[col]

        self.df_to_return = self.df_to_return.merge(temp_df_rn1, on=self.columns_to_group, how='left')
        self.df_to_return = self.df_to_return.merge(temp_df_rn_1, on=self.columns_to_group, how='left')
        self.df_to_return = self.df_to_return.merge(temp_df_agg_sum, on=self.columns_to_group, how='left')
        self.df_to_return = self.df_to_return.merge(temp_df_agg_count, on=self.columns_to_group, how='left')

    def merge_aggregation_by_period(self):
        pass
