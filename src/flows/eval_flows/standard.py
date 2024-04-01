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
    """
    A class that inherits from the `SessionManager` class and performs evaluation tasks.

    Attributes:
    - args (object): The arguments passed to the class.
    - models_list (list): A list of models to be evaluated.
    - output_df (pandas.DataFrame): The output DataFrame.

    Methods:
    - __init__(self, args):
        - Initializes the class by calling the parent class's `__init__` method.
        - Checks if there are multiclass artifacts from training sessions and sets the `models_list` accordingly.
        - Assigns the `args` parameter to the `args` attribute.
        - Initializes the `output_df` attribute to `None`.
    - run(self):
        - Orchestrates the evaluation process.
        - Creates an evaluation session folder.
        - Iterates through the models in the `models_list` and calls the `merge_word` function for each model.
        - Prints the end of the evaluation process.

    """
    def __init__(self, args):
        SessionManager.__init__(self, args)
        #self.main_model = args.main_model.lower()

        # check if we have multiclass artifacts from training sessions
        if os.path.exists(f"output_data/{self.project_name}/ppscore.csv"):
            self.models_list = ['xgb']
        else:
            self.models_list = ['xgb', 'rf', 'dt']
        
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
            merge_word(project_name=self.project_name,input_data_folder_name=self.input_data_folder_name,
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
        """
        Creates an evaluation session folder.

        Steps:
        1. Print and log a message indicating the start of the session folder creation.
        2. Generate the session ID by concatenating 'EVAL_', the input data project folder, the start time, and the tag.
        3. Set the session ID folder path by concatenating the session folder name and the session ID.
        4. Create the session ID folder using the `os.mkdir()` function.
        5. Print and log a message indicating the completion of the session folder creation.

        Returns:
        None
        """
        print_and_log('Createing session folder. Starting', 'YELLOW')
        self.session_id = 'EVAL_' + self.input_data_project_folder + '_' + str(self.start_time) + '_' + self.tag
        self.session_id_folder = self.session_folder_name + self.session_id
        os.mkdir(self.session_id_folder)
        print_and_log('Createing session folder. Done', '')