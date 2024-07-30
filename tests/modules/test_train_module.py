"""Tests data preparation"""
import pandas as pd

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
    def test_train_module(self):
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

        module = ModuleClass(args=args, production_or_test="test")
        module.prepare()

        try:
            module.run()
        except Exception as e:
            print_and_log(e, "RED")

        # Example usage:
        base_directory = "../../sessions"
        latest_session_dir = self.get_latest_session_folder(base_directory)

        self.unittest_data = CrossValDataLoader(data_dir="../unittest/train_output_data")
        self.input_data = CrossValDataLoader(data_dir=latest_session_dir)

        print("Nice")
    def test_values(self):

        return None

    def test_shapes_and_dtypes(self):
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