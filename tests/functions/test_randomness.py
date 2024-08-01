from pickle import dump
from pickle import load

import pandas as pd
import json
import inspect

from datetime import datetime
import ppscore as pps

from src.classes.DfAggregator import DfAggregator
from src.classes.OptimalBinning import OptimaBinning
from src.classes.SessionManager import SessionManager

import src.functions.data_prep.dates_manipulation as date_funcs
from src.functions.data_prep.misc_functions import split_columns_by_types, switch_numerical_to_object_column, \
    convert_obj_to_cat_and_get_dummies, remove_column_if_not_in_final_features, \
    create_dict_based_on_col_name_contains, create_ratios, correlation_matrix, \
    remove_categorical_cols_with_too_many_values, treating_outliers,\
    nan_inspector, create_checkpoint, compare_checkpoints

def compare_pipeline_runs(run1_checkpoints, run2_checkpoints):
    for cp1, cp2 in zip(run1_checkpoints, run2_checkpoints):
        compare_checkpoints(cp1, cp2)

run1_checkpoints_dir = "../../run1.json"
run2_checkpoints_dir = "../../run2.json"

with open(run1_checkpoints_dir, 'r') as json_file:
    run1_checkpoints = json.load(json_file)

with open(run2_checkpoints_dir, 'r') as json_file:
    run2_checkpoints = json.load(json_file)


compare_pipeline_runs(run1_checkpoints, run2_checkpoints)

from prettytable import PrettyTable
import json


def compare_dicts(dict1, dict2):
    all_keys = set(dict1.keys()) | set(dict2.keys())
    table = PrettyTable()
    table.field_names = ["Key", "Run 1 Value", "Run 2 Value"]
    table.align["Key"] = "l"
    table.align["Run 1 Value"] = "l"
    table.align["Run 2 Value"] = "l"

    for key in sorted(all_keys):
        value1 = json.dumps(dict1.get(key, "N/A"), indent=2)
        value2 = json.dumps(dict2.get(key, "N/A"), indent=2)
        if value1 != value2:
            value1 = f"\033[91m{value1}\033[0m"  # Red color for differences
            value2 = f"\033[91m{value2}\033[0m"
        table.add_row([key, value1, value2])

    print(table)


# Your existing code to load the data
with open(run1_checkpoints_dir, 'r') as json_file:
    run1_checkpoints = json.load(json_file)

with open(run2_checkpoints_dir, 'r') as json_file:
    run2_checkpoints = json.load(json_file)

for i in range(len(run1_checkpoints)):
    # Compare the 9th dictionary from each run
    compare_dicts(run1_checkpoints[i], run2_checkpoints[i])