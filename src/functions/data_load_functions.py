import logging
import sys
from os import listdir
from os.path import isfile, join

import pandas as pd
from src.functions.printing_and_logging import print_and_log


def check_files_for_csv_suffix(input_data_folder_name, input_data_project_folder):
    """
    Check the files in the input folder for a CSV suffix and return the CSV file.

    Parameters:
    - input_data_folder_name: str, name of the input data folder
    - input_data_project_folder: str, name of the project folder within the input data folder

    Returns:
    - input_file: str, name of the CSV file in the input folder
    """
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
    """
    Check the separator of the CSV file and handle errors if the separator is not ',' or ';'.

    Parameters:
    - input_data_folder_name: str, name of the input data folder
    - input_data_project_folder: str, name of the project folder within the input data folder
    - input_df: DataFrame, input data DataFrame
    - input_file: str, name of the CSV file in the input folder

    Returns:
    - input_df: DataFrame, updated input data DataFrame
    """
    if len(input_df.columns.to_list()) == 1:
        input_df = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file,
                               sep=';')
        if len(input_df.columns.to_list()) == 1:
            print_and_log('ERROR: input data separator not any of the following ,;', 'RED')
            sys.exit()
    return input_df


