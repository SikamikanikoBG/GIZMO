"""Tests data preparation"""
import argparse
import unittest
import sys
import os
from tests.cross_validation import CrossValDataLoader
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)

from src.flows.data_prep_flows.standard import ModuleClass


class CrossValDataLoader:
    def __init__(self, data_dir: str):
        """
        CrossValDataLoader - A class for loading data from a directory into Pandas DataFrames.

        This class streamlines the process of loading multiple files from a single directory into
        Pandas DataFrames for cross-validation or other data analysis tasks. It supports various file
        formats and dynamically sets DataFrame attributes based on file names.

        Attributes:
            data_dir (str): The path to the directory containing the data files.

        Methods:
            load_files(): Loads all supported files (CSV, PKL, Feather, Parquet) into Pandas DataFrames.
                         DataFrames are named after their filenames (without extension) and set as class attributes.
        """

        self.data_dir = data_dir

        self.load_files()

    def load_files(self):
        for filename in os.listdir(self.data_dir):
            name, extension = os.path.splitext(filename)
            file_path = os.path.join(self.data_dir, filename)

            try:
                if extension == '.csv':
                    setattr(self, name, pd.read_csv(file_path))
                elif extension == '.pkl':
                    setattr(self, name, pd.read_pickle(file_path))
                elif extension == '.feather':
                    setattr(self, name, pd.read_feather(file_path))
                elif extension == '.parquet':
                    setattr(self, name, pd.read_parquet(file_path))
                else:
                    print(f"Skipping unsupported file: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")


class TestDataPrep(unittest.TestCase):
    """Tests data preparation"""

    def test_data_prep(self):
        """Tests data preparation"""
        args = argparse.Namespace(data_prep_module='standard', project='bg_stage2',
                                  session=None, model=None, tag=None, predict_module=None)
        module = ModuleClass(args=args, production_or_test="test")

        module.prepare()
        module.run()

        expected_output_data = CrossValDataLoader(
            "../../unittest/data_prep_output_data")
        output_data = CrossValDataLoader("../../../output_data/bg_stage2")

        def compare(column_name):
            print(expected_output_data.__getattribute__(column_name))
            first_column = expected_output_data.__getattribute__(column_name).columns[0]
            df_expected_output_data_sorted = expected_output_data.__getattribute__(column_name).sort_values(
                by=first_column).reset_index(drop=True)
            df_output_data_sorted = output_data.__getattribute__(column_name).sort_values(
                by=first_column).reset_index(drop=True)

            print(df_expected_output_data_sorted.compare(
                df_output_data_sorted))
            self.assertTrue(df_expected_output_data_sorted.equals(
                df_output_data_sorted), f"The {column_name} data does not match with the expected")

        compare("correl")
        compare("final_features")
        compare("missing_values")
        compare("output_data_file_full")
        compare("output_data_file")


if __name__ == '__main__':
    unittest.main()
