import os
import pandas as pd
# import pickle
from pyarrow import parquet as pq

class CrossValDataLoader():

    def __init__(self, data_dir: str):
        self.correl = pd.read_csv(os.path.join(data_dir, "correl.csv"))
        self.final_features = pd.read_pickle(os.path.join(data_dir, "final_features.pkl"))
        self.missing_values = pd.read_csv(os.path.join(data_dir, "missing_values.csv"))
        self.output_data_file = pd.read_parquet(os.path.join(data_dir, "output_data_file.parquet"))
        self.output_data_file_full = pd.read_parquet(os.path.join(data_dir, "output_data_file_full.parquet"))

# def parse_input_data(test_input_data_root_dir: str) -> list:
#     dirs = os.listdir(test_input_data_root_dir)
#     dirs = [os.path.join(test_input_data_root_dir, dir) for dir in dirs]
#     return dirs

def get_data_dir(root_dir: str) -> str:
    return os.path.abspath(root_dir)

def run_cv():
    test_data_dir = get_data_dir("./test_data")
    input_data_dir = get_data_dir("../output_data/bg_stage2/")

    test_loader = CrossValDataLoader(test_data_dir)
    input_loader = CrossValDataLoader(input_data_dir)

    compare_data(test_loader, input_loader)
    print("Success!")

def compare_data(test_loader: CrossValDataLoader, input_loader: CrossValDataLoader):
    assert test_loader.correl.equals(input_loader.correl), "Correlation does not match"
    assert test_loader.final_features == input_loader.final_features, "Final features do not match"
    assert test_loader.missing_values.equals(input_loader.missing_values), "Missing values table does not match"
    assert test_loader.output_data_file.equals(input_loader.output_data_file), "Output data does not match"
    assert test_loader.output_data_file_full.equals(input_loader.output_data_file_full), "Full output data does not match"


run_cv()