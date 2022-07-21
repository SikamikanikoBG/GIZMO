import logging
import sys
from os import listdir
from os.path import isfile, join

import pandas as pd
from src.functions.printing_and_logging_functions import print_and_log


def check_files_for_csv_suffix(input_data_folder_name, input_data_project_folder):
    only_files = [f for f in listdir(input_data_folder_name + input_data_project_folder + '/') if
                  isfile(join(input_data_folder_name + input_data_project_folder + '/', f))]
    if len(only_files) == 0:
        print_and_log('ERROR: No files in input folder. Aborting the program.', 'RED')
        sys.exit()
    if 'dict' in only_files[0]:
        input_file = only_files[1]
    else:
        input_file = only_files[0]
    _, _, suffix = str(input_file).partition('.')
    if 'csv' not in suffix:
        print_and_log('ERROR: input data not a csv file.', 'RED')
        sys.exit()
    return input_file


def check_separator_csv_file(input_data_folder_name, input_data_project_folder, input_df, input_file):
    if len(input_df.columns.to_list()) == 1:
        input_df = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file,
                               sep=';')
        if len(input_df.columns.to_list()) == 1:
            print_and_log('ERROR: input data separator not any of the following ,;', 'RED')
            sys.exit()
    return input_df


