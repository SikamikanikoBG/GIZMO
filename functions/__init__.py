from .data_cleaning import missing_values, outliar, split_columns_by_types, convert_category, data_load, data_cleaning
from .modeller import xgb, cut_into_bands, rand_forest, decision_tree, raw_features_to_list, lr_run
from .feature_engineering import create_ratios, create_tree_feats, correlation_matrix
from .ascii_art import print_load, print_end, print_train, print_eval
from .evaluation import merge_word


class Session:
    def __init__(self):
        self.session_id = None
        self.params = None


class Loader:
    def __init__(self):
        self.input_df, self.input_df_full = None, None



