"""Tests data preparation"""
import argparse
import unittest
import sys
import os
from src.functions.printing_and_logging import print_and_log

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)

from src.flows.training_flows.standard import ModuleClass

class TestTrainModule(unittest.TestCase):
    def test_train_module(self):
        args = argparse.Namespace(
            project="bg_stage2",
            train_module="standard",
            tag=None,  # Optional
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

        module = ModuleClass(args=args)
        module.prepare()

        try:
            module.run()
        except Exception as e:
            print_and_log(e, "RED")


if __name__ == '__main__':
    unittest.main()