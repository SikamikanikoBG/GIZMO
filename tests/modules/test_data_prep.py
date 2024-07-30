"""Tests data preparation"""
import argparse
import unittest
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)

from src.flows.data_prep_flows.standard import ModuleClass

class TestDataPrep(unittest.TestCase):
    """Tests data preparation"""
    def test_data_prep(self):
        # setting path

        args = argparse.Namespace(data_prep_module='standard', project='bg_stage2', session=None, model=None, tag=None, predict_module=None)
        #module_lib = import_module(package="../../src", name=f'src.flows.data_prep_flows.{args.data_prep_module}')
        module = ModuleClass(args=args)
        module.prepare()
        module.run()


if __name__ == '__main__':
    unittest.main()