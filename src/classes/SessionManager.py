import json
import logging
import os
from datetime import datetime

from colorama import Fore, Style

import definitions
from src import BaseLoader, print_and_log


class SessionManager:
    def __init__(self, args):
        self.check4_time_runtime = None
        self.check3_time_runtime = None
        self.check2_time_runtime = None
        self.check1_time_runtime = None
        self.start_time = datetime.now()  # .strftime("%Y-%m-%d_%H:%M:%S")
        self.end_time = None
        self.run_time = None
        self.data_load_time = None
        self.check1_time = None
        self.check2_time = None
        self.check3_time = None
        self.check4_time = None
        self.session_id = None
        self.params = None
        self.args = args
        self.session_to_eval = self.args.session
        self.model_arg = self.args.model
        # self.run_info = self.args.run
        self.project_name = self.args.project
        if not self.args.tag:
            self.tag = 'no_tag'
        else:
            self.tag = self.args.tag

        # folder structure properties
        if self.args.data_prep_module:
            self.log_file_name = 'load_' + self.args.data_prep_module + '_' + self.project_name + '_' + str(
                self.start_time) + '_' + self.tag + ".log"
        elif self.args.train_module:
            self.log_file_name = 'train_' + self.args.train_module + '_' + self.project_name + '_' + str(
                self.start_time) + '_' + self.tag + ".log"
        elif self.args.predict_module:
            self.log_file_name = 'predict_' + self.args.predict_module + '_' + self.project_name + '_' + str(
                self.start_time) + '_' + self.tag + ".log"
        self.log_folder_name = definitions.ROOT_DIR + '/logs/'
        self.session_folder_name = definitions.ROOT_DIR + '/sessions/'
        self.session_id_folder = None
        self.input_data_folder_name = definitions.ROOT_DIR + '/input_data/'
        self.input_data_project_folder = self.project_name
        self.output_data_folder_name = definitions.ROOT_DIR + '/output_data/'
        self.functions_folder_name = definitions.ROOT_DIR + '/src/'
        self.params_folder_name = definitions.ROOT_DIR + '/params/'
        self.implemented_folder = definitions.ROOT_DIR + '/implemented_models/'

        # Import parameters
        try:
            with open(definitions.ROOT_DIR + '/params/params_' + self.project_name + '.json') as json_file:
                self.params = json.load(json_file)
            self.criterion_column = self.params['criterion_column']
            self.missing_treatment = self.params["missing_treatment"]
            self.observation_date_column = self.params["observation_date_column"]
            self.columns_to_exclude = self.params["columns_to_exclude"]
            self.periods_to_exclude = self.params["periods_to_exclude"]
            self.t1df_period = self.params["t1df"]
            self.t2df_period = self.params["t2df"]
            self.t3df_period = self.params["t3df"]
            self.lr_features = self.params["lr_features"]
            self.cut_offs = self.params["cut_offs"]
            self.under_sampling = self.params['under_sampling']
            self.optimal_binning_columns = self.params['optimal_binning_columns']
            self.main_table = self.params["main_table"]
        except Exception as e:
            print(Fore.RED + 'ERROR: params file not available' + Style.RESET_ALL)
            print(e)
            logging.error(e)
            quit()

        self.loader = BaseLoader(params=self.params)
        if self.args.data_prep_module:
            self.loader.data_load_prep(in_data_folder=self.input_data_folder_name,
                                       in_data_proj_folder=self.input_data_project_folder)
        elif self.args.train_module:
            self.loader.data_load_train(output_data_folder_name=self.output_data_folder_name,
                                        input_data_project_folder=self.input_data_project_folder)
        elif self.args.predict_module:
            pass

    def prepare(self):
        """
        Orchestrates the preparation of the session run
        Returns:

        """
        self.create_folders()

    def create_folders(self):
        """
        Creates the folder structure if some of it is missing
        Returns:

        """
        if not os.path.isdir(self.log_folder_name):
            os.mkdir(self.log_folder_name)
        if not os.path.isdir(self.session_folder_name):
            os.mkdir(self.session_folder_name)
        if not os.path.isdir(self.input_data_folder_name):
            os.mkdir(self.input_data_folder_name)
        if not os.path.isdir(self.output_data_folder_name):
            os.mkdir(self.output_data_folder_name)
        if not os.path.isdir(self.functions_folder_name):
            os.mkdir(self.functions_folder_name)
        if not os.path.isdir(self.params_folder_name):
            os.mkdir(self.params_folder_name)
        if not os.path.isdir(self.implemented_folder):
            os.mkdir(self.implemented_folder)

        # Create output data project folder
        if not os.path.isdir(self.output_data_folder_name + self.input_data_project_folder + '/'):
            os.mkdir(self.output_data_folder_name + self.input_data_project_folder + '/')

    def start_logging(self):
        """
        Starts the logging process for the session and creates the log file
        Returns:

        """
        # logging
        if not os.path.isfile(self.log_folder_name + self.log_file_name):
            open(self.log_folder_name + self.log_file_name, 'w').close()

        logging.basicConfig(
            filename=self.log_folder_name + self.log_file_name,
            level=logging.INFO, format='%(asctime)s - %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        print_and_log('[ LOGGING ] Start logging', 'YELLOW')

    def run_time_calc(self):
        """
        Calculates the time delta that the session took to run
        Returns:

        """
        self.end_time = datetime.now()
        self.run_time = round(float((self.end_time - self.start_time).total_seconds()), 2)
        if self.check1_time:
            self.check1_time_runtime = round(float((self.check1_time - self.start_time).total_seconds()), 2)
        if self.check2_time:
            self.check2_time_runtime = round(float((self.check2_time - self.check1_time).total_seconds()), 2)
        if self.check3_time:
            self.check3_time_runtime = round(float((self.check3_time - self.check2_time).total_seconds()), 2)
        if self.check4_time:
            self.check4_time_runtime = round(float((self.check4_time - self.check3_time).total_seconds()), 2)

        print_and_log(f"RUN time: {self.run_time}, "
                      f"Check1 time: {self.check1_time_runtime}c, "
                      f"Check2 time: {self.check2_time_runtime}c, "
                      f"Check3 time: {self.check3_time_runtime}c, "
                      f"Check4 time: {self.check4_time_runtime}c, ", "YELLOW")
