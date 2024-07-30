"""Tests data preparation"""
import pandas as pd
import numpy as np
import argparse
import unittest
import glob
import sys
import os

from src.flows.training_flows.standard import ModuleClass
from src.functions.printing_and_logging import print_and_log
from tests.cross_validation import CrossValDataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)

class TestTrainModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This runs once for the whole class
        args = argparse.Namespace(
            project="bg_stage2",
            train_module="standard",
            tag=None,
            session=None,
            model=None,
            data_prep_module=None,
            predict_module=None,
            pred_data_prep=None,
            eval_module=None,
            nb_tree_features=None,
            main_model=None,
            h=None,
        )
        cls.module = ModuleClass(args=args, production_or_test="test")
        cls.module.prepare()

    def setUp(self):
        # This runs before each test method
        base_directory = "../../sessions"
        latest_session_dir = self.get_latest_session_folder(base_directory)

        self.unittest_data = CrossValDataLoader(data_dir="../unittest/train_output_data")
        self.input_data = CrossValDataLoader(data_dir=latest_session_dir)

    def test_train_module(self):
        try:
            self.module.run()
        except Exception as e:
            print_and_log(e, "RED")
            self.fail(f"ModuleClass.run() raised {type(e).__name__} unexpectedly!")

    def test_values(self):
        """Test for exact values"""
        for attr in dir(self.unittest_data):
            if not attr.startswith('__') and isinstance(getattr(self.unittest_data, attr), pd.DataFrame):
                unittest_df = getattr(self.unittest_data, attr)
                input_df = getattr(self.input_data, attr, None)

                self.assertIsNotNone(input_df, f"DataFrame {attr} not found in input_data")

                pd.testing.assert_frame_equal(unittest_df, input_df,
                                              check_dtype=False,
                                              check_index_type=False,
                                              check_column_type=False,
                                              check_frame_type=False)
                print(f"Values in {attr} match between unittest_data and input_data")

    def test_shapes_and_dtypes(self):
        """Test for shapes"""
        for attr in dir(self.unittest_data):
            if not attr.startswith('__') and isinstance(getattr(self.unittest_data, attr), pd.DataFrame):
                unittest_df = getattr(self.unittest_data, attr)
                input_df = getattr(self.input_data, attr, None)

                self.assertIsNotNone(input_df, f"DataFrame {attr} not found in input_data")

                self.assertEqual(unittest_df.shape, input_df.shape,
                                 f"Shapes do not match for {attr}: {unittest_df.shape} vs {input_df.shape}")

                self.assertTrue(unittest_df.dtypes.equals(input_df.dtypes),
                                f"Data types do not match for {attr}")

                print(f"Shapes and dtypes in {attr} match between unittest_data and input_data")
        return None

    def test_missing_values(self):
        """Test for missing values"""
        for attr in dir(self.unittest_data):
            if not attr.startswith('__') and isinstance(getattr(self.unittest_data, attr), pd.DataFrame):
                unittest_df = getattr(self.unittest_data, attr)
                input_df = getattr(self.input_data, attr, None)

                self.assertIsNotNone(input_df, f"DataFrame {attr} not found in input_data")

                unittest_missing = unittest_df.isnull().sum()
                input_missing = input_df.isnull().sum()

                pd.testing.assert_series_equal(unittest_missing, input_missing,
                                               check_names=False)
                print(f"Missing value counts in {attr} match between unittest_data and input_data")

    def test_column_names(self):
        """Test for column names"""
        for attr in dir(self.unittest_data):
            if not attr.startswith('__') and isinstance(getattr(self.unittest_data, attr), pd.DataFrame):
                unittest_df = getattr(self.unittest_data, attr)
                input_df = getattr(self.input_data, attr, None)

                self.assertIsNotNone(input_df, f"DataFrame {attr} not found in input_data")

                self.assertListEqual(list(unittest_df.columns), list(input_df.columns),
                                     f"Column names do not match for {attr}")
                print(f"Column names in {attr} match between unittest_data and input_data")

    def test_data_ranges(self):
        """Test for data ranges (min, max, mean)"""
        for attr in dir(self.unittest_data):
            if not attr.startswith('__') and isinstance(getattr(self.unittest_data, attr), pd.DataFrame):
                unittest_df = getattr(self.unittest_data, attr)
                input_df = getattr(self.input_data, attr, None)

                self.assertIsNotNone(input_df, f"DataFrame {attr} not found in input_data")

                for col in unittest_df.select_dtypes(include=[np.number]).columns:
                    self.assertAlmostEqual(unittest_df[col].min(), input_df[col].min(),
                                           msg=f"Minimum values don't match for {attr}.{col}")
                    self.assertAlmostEqual(unittest_df[col].max(), input_df[col].max(),
                                           msg=f"Maximum values don't match for {attr}.{col}")
                    self.assertAlmostEqual(unittest_df[col].mean(), input_df[col].mean(),
                                           msg=f"Mean values don't match for {attr}.{col}")
                print(f"Data ranges in {attr} match between unittest_data and input_data")

    def test_categorical_uniques(self):
        """Test for unique values in categorical columns"""
        for attr in dir(self.unittest_data):
            if not attr.startswith('__') and isinstance(getattr(self.unittest_data, attr), pd.DataFrame):
                unittest_df = getattr(self.unittest_data, attr)
                input_df = getattr(self.input_data, attr, None)

                self.assertIsNotNone(input_df, f"DataFrame {attr} not found in input_data")

                for col in unittest_df.select_dtypes(include=['object', 'category']).columns:
                    unittest_uniques = set(unittest_df[col].unique())
                    input_uniques = set(input_df[col].unique())
                    self.assertSetEqual(unittest_uniques, input_uniques,
                                        f"Unique values don't match for {attr}.{col}")
                print(f"Categorical unique values in {attr} match between unittest_data and input_data")
    def get_latest_session_folder(self, base_dir):
        # Pattern to match session folders
        pattern = os.path.join(base_dir, "TRAIN_bg_stage2_*_no_tag")

        # Get all matching directories
        sessions = glob.glob(pattern)

        if not sessions:
            return None  # No matching sessions found

        # Sort sessions by creation time (most recent first)
        latest_session = max(sessions, key=os.path.getctime)

        return latest_session

if __name__ == '__main__':
    unittest.main()