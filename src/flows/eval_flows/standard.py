import importlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

import definitions
from src.classes.SessionManager import SessionManager
from src.functions import api_communication
from src.functions.predict.calcula_data_drift import calculate_data_drift
from src.functions.printing_and_logging import print_end, print_and_log
from src.functions.evaluation import merge_word


class ModuleClass(SessionManager):
    def __init__(self, args):
        SessionManager.__init__(self, args)
        #self.main_model = args.main_model.lower()
        self.models_list = ['xgb']
        # self.models_list = ['xgb', 'rf', 'dt']
        # self.models_list = ['xgb']
        self.args = args
        self.output_df = None

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        # generate doxc
        print("[ EVAL ] START")
        self.create_eval_session_folder()

        for model in self.models_list:
            print(f"[ EVAL ] Model: {model}")
            merge_word(input_data_folder_name=self.input_data_folder_name,
                   input_data_project_folder=self.input_data_project_folder,
                   session_to_eval=self.session_to_eval,
                   session_folder_name = self.session_folder_name,
               session_id_folder=self.session_id_folder,
                   criterion_column=self.criterion_column,
               observation_date_column=self.observation_date_column,
               columns_to_exclude=self.columns_to_exclude,
               periods_to_exclude=self.periods_to_exclude,
               t1df_period=self.t1df_period,
               t2df_period=self.t2df_period,
               t3df_period=self.t3df_period,
               model_arg=model,
               missing_treatment = 1
                   , params=self.params)

        print_end()

    def create_eval_session_folder(self):
        print_and_log('Createing session folder. Starting', 'YELLOW')
        self.session_id = 'EVAL_' + self.input_data_project_folder + '_' + str(self.start_time) + '_' + self.tag
        self.session_id_folder = self.session_folder_name + self.session_id
        os.mkdir(self.session_id_folder)
        print_and_log('Createing session folder. Done', '')