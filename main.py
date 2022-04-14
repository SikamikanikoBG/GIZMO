import logging
import os
import sys
import pandas as pd
from datetime import datetime
import json
import pickle
import argparse
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import warnings
import colorama
from colorama import Fore
from colorama import Style


import functions

warnings.filterwarnings("ignore")

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('--run', type=str, help='After run write which command to execute: load, train, eval')
parser.add_argument('--tag', type=str, help='Tag the training session. Optional')
parser.add_argument('--project', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
parser.add_argument('--session', type=str, help='Train session folder to generate evaluation docx')
parser.add_argument('--model', type=str, help='Model to evaluate. xgb, rf, dt, lr')
parser.add_argument('--h', type=str, help='You need help...')
args = parser.parse_args()

help_need = args.h
if help_need:
    print(parser.parse_known_args())
    sys.exit()

session_to_eval = args.session
model_arg = args.model
run_info = args.run
project_name = args.project
if not args.tag:
    tag = 'no_tag'
else:
    tag = args.tag

# Creating folder structure
start = datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
log_file_name = run_info + '_' + project_name + '_' + str(start) + '_' + tag + ".log"
log_folder_name = './log/'
session_folder_name = './sessions/'
input_data_folder_name = './input_data/'
input_data_project_folder = project_name
output_data_folder_name = './output_data/'
functions_folder_name = './functions/'
params_folder_name = './params/'

if not os.path.isdir(log_folder_name):
    os.mkdir(log_folder_name)
if not os.path.isdir(session_folder_name):
    os.mkdir(session_folder_name)
if not os.path.isdir(input_data_folder_name):
    os.mkdir(input_data_folder_name)
if not os.path.isdir(output_data_folder_name):
    os.mkdir(output_data_folder_name)
if not os.path.isdir(functions_folder_name):
    os.mkdir(functions_folder_name)
if not os.path.isdir(params_folder_name):
    os.mkdir(params_folder_name)
# Create outputdata project folder
if not os.path.isdir(output_data_folder_name + input_data_project_folder + '/'):
    os.mkdir(output_data_folder_name + input_data_project_folder + '/')

# logging
if not os.path.isfile(log_folder_name + log_file_name):
    open(log_folder_name + log_file_name, 'w').close()

logging.basicConfig(
    filename=log_folder_name + log_file_name,
    level=logging.INFO, format='%(asctime)s - %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Starting')

# Import parameters
try:
    with open('params/params_' + project_name + '.json') as json_file:
        params = json.load(json_file)
    criterion_column = params['criterion_column']
    missing_treatment = params["missing_treatment"]
    observation_date_column = params["observation_date_column"]
    columns_to_exclude = params["columns_to_exclude"]
    periods_to_exclude = params["periods_to_exclude"]
    t1df_period = params["t1df"]
    t2df_period = params["t2df"]
    t3df_period = params["t3df"]
    lr_features = params["lr_features"]
    cut_offs = params["cut_offs"]
except:
    print(Fore.RED + 'ERROR: params file not available' + Style.RESET_ALL)
    pass


def modeller(input_data_project_folder):
    print(Fore.YELLOW + '\n Loading data \n' + Style.RESET_ALL)
    logging.info('Loading data')
    input_df = pq.read_table(output_data_folder_name + input_data_project_folder + '/' 'output_data_file.parquet')
    input_df = input_df.to_pandas()
    # checking for duplicated columns and treating them
    input_df = input_df.loc[:, ~input_df.columns.duplicated()]

    if params['under_sampling']:
        input_df_full = pq.read_table(
            output_data_folder_name + input_data_project_folder + '/' 'output_data_file_full.parquet')
        input_df_full = input_df_full.to_pandas()
        input_df_full = input_df_full.loc[:, ~input_df_full.columns.duplicated()]

    print('\t Data Loaded')
    logging.info("Data Loaded")
    with (open(output_data_folder_name + input_data_project_folder + '/' + 'final_features.pkl', "rb")) as openfile:
        final_features = pickle.load(openfile)
    print(f'\t All observation dates before splitting the file: {observation_date_column}: {input_df[observation_date_column].unique()}')
    if params['under_sampling']:
        print('\n Splitting temporal validation dataframes\n ', t1df_period, t2df_period, t3df_period)
        logging.info(f"Splitting temporal validation dataframes, {t1df_period}, {t2df_period}, {t3df_period}")
        t1df = input_df_full[input_df_full[observation_date_column] == t1df_period]
        print('\t t1 done. Shape: ', len(t1df))
        logging.info(f"t1 done. Shape: ', {len(t1df)}")
        t2df = input_df_full[input_df_full[observation_date_column] == t2df_period]
        print('\t t2 done. Shape: ', len(t2df))
        logging.info(f"t2 done. Shape: ', {len(t2df)}")
        t3df = input_df_full[input_df_full[observation_date_column] == t3df_period]
        print('\t t3 done. Shape: ', len(t3df))
        logging.info(f"t3 done. Shape: ', {len(t3df)}")
    else:
        print('\t Splitting temporal validation dataframes', t1df_period, t2df_period, t3df_period)
        logging.info(f"Splitting temporal validation dataframes, {t1df_period}, {t2df_period}, {t3df_period}")
        t1df = input_df[input_df[observation_date_column] == t1df_period]
        print('\t t1 done. Shape: ', len(t1df))
        logging.info(f"t1 done. Shape: ', {len(t1df)}")
        t2df = input_df[input_df[observation_date_column] == t2df_period]
        print('\t t2 done. Shape: ', len(t2df))
        logging.info(f"t2 done. Shape: ', {len(t2df)}")
        t3df = input_df[input_df[observation_date_column] == t3df_period]
        print('\t t3 done. Shape: ', len(t3df))
        logging.info(f"t3 done. Shape: ', {len(t3df)}")

    input_df = input_df[input_df[observation_date_column] != t1df_period]
    input_df = input_df[input_df[observation_date_column] != t2df_period]
    input_df = input_df[input_df[observation_date_column] != t3df_period]

    if params['under_sampling']:
        input_df_full = input_df_full[input_df_full[observation_date_column] != t1df_period]
        input_df_full = input_df_full[input_df_full[observation_date_column] != t2df_period]
        input_df_full = input_df_full[input_df_full[observation_date_column] != t3df_period]

    print('\n Splitting temporal validation dataframes. Done')
    logging.info("Splitting temporal validation dataframes. Done")

    print(Fore.YELLOW + '\n Splitting train and test dataframes. Starting' + Style.RESET_ALL)
    logging.info("Splitting train and test dataframes. Starting")

    if params['under_sampling']:
        X_train, X_test, y_train, y_test = train_test_split(
            input_df_full, input_df_full[criterion_column], test_size=0.33, random_state=42)
        X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(
            input_df, input_df[criterion_column], test_size=0.33, random_state=42)
        del input_df
        del input_df_full
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            input_df, input_df[criterion_column], test_size=0.33, random_state=42)
        X_train_us = pd.DataFrame()
        X_test_us = pd.DataFrame()
        y_train_us = pd.DataFrame()
        y_test_us = pd.DataFrame()
        del input_df

    print('\t Splitting train and test dataframes. Done')
    logging.info('Splitting train and test dataframes. Done')
    models = pd.DataFrame(columns=['Method', 'AccuracyScore', 'AUC', 'PrecisionScore', 'Recall', 'F1', 'DataSet', 'NbFeatures'])

    print(Fore.YELLOW + '\n Createing session folder. Starting\n' + Style.RESET_ALL)
    logging.info('Createing session folder. Starting')
    session_id = 'TRAIN_' + input_data_project_folder + '_' + str(start) + '_' + tag
    session_id_folder = session_folder_name + session_id
    os.mkdir(session_id_folder)
    os.mkdir(session_id_folder + '/dt/')
    os.mkdir(session_id_folder + '/rf/')
    os.mkdir(session_id_folder + '/xgb/')
    os.mkdir(session_id_folder + '/lr/')
    print(f'\t Createing session folder. Done - {session_id_folder}')
    logging.info('Createing session folder. Done')

    # XGB
    print(Fore.YELLOW + '\n *** Starting XGB modeller *** \n' + Style.RESET_ALL)
    logging.info('\n *** Starting XGB modeller ***')
    X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, cut_points_train, x_train_recall, x_train_f1 = functions.xgb(
        df=X_train, criterion=y_train,
        df_us=X_train_us, criterion_us=y_train_us,
        test_X_us=X_test_us, test_y_us=y_test_us,
        model_to_predict=None,
        predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features, cut_points_train=None,
        cut_offs=cut_offs, params=params)
    X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, _, x_test_recall, x_test_f1 = functions.xgb(df=X_test, criterion=y_test,
                                                                                            df_us=X_train_us,
                                                                                            criterion_us=y_train_us,
                                                                                            test_X_us=X_test_us,
                                                                                            test_y_us=y_test_us,
                                                                                            model_to_predict=model_train,
                                                                                            predict_only_flag='yes',
                                                                                            test_X=None, test_y=None,
                                                                                            final_features=final_features,
                                                                                            cut_points_train=cut_points_train,
                                                                                            cut_offs=cut_offs,
                                                                                            params=params)
    t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, _, t1df_recall, t1df_f1 = functions.xgb(df=t1df,
                                                                                  criterion=t1df[criterion_column],
                                                                                  df_us=X_train_us,
                                                                                  criterion_us=y_train_us,
                                                                                  test_X_us=X_test_us,
                                                                                  test_y_us=y_test_us,
                                                                                  model_to_predict=model_train,
                                                                                  predict_only_flag='yes', test_X=None,
                                                                                  test_y=None,
                                                                                  final_features=final_features,
                                                                                  cut_points_train=cut_points_train,
                                                                                  cut_offs=cut_offs, params=params)
    t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, _, t2df_recall, t2df_f1 = functions.xgb(df=t2df,
                                                                                  criterion=t2df[criterion_column],
                                                                                  df_us=X_train_us,
                                                                                  criterion_us=y_train_us,
                                                                                  test_X_us=X_test_us,
                                                                                  test_y_us=y_test_us,
                                                                                  model_to_predict=model_train,
                                                                                  predict_only_flag='yes', test_X=None,
                                                                                  test_y=None,
                                                                                  final_features=final_features,
                                                                                  cut_points_train=cut_points_train,
                                                                                  cut_offs=cut_offs, params=params)
    t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, _, t3df_recall, t3df_f1 = functions.xgb(df=t3df,
                                                                                  criterion=t3df[criterion_column],
                                                                                  df_us=X_train_us,
                                                                                  criterion_us=y_train_us,
                                                                                  test_X_us=X_test_us,
                                                                                  test_y_us=y_test_us,
                                                                                  model_to_predict=model_train,
                                                                                  predict_only_flag='yes', test_X=None,
                                                                                  test_y=None,
                                                                                  final_features=final_features,
                                                                                  cut_points_train=cut_points_train,
                                                                                  cut_offs=cut_offs, params=params)

    models = models.append(
        {'Method': 'xgb', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec, 'Recall' : x_train_recall, 'F1': x_train_f1,
         'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'xgb', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec, 'Recall' : x_test_recall, 'F1': x_test_f1,
         'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'xgb', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec, 'Recall' : t1df_recall, 'F1': t1df_f1,
         'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'xgb', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec, 'Recall' : t2df_recall, 'F1': t2df_f1,
         'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'xgb', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec,  'Recall' : t3df_recall, 'F1': t3df_f1,
         'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
        ignore_index=True)
    pickle.dump(model_train, open(session_id_folder + '/xgb/model_train.pkl', 'wb'))
    feature_importance.to_csv(session_id_folder + '/xgb/feat_importance.csv', index=False)
    print(f"\t Feature importance: {feature_importance}")
    logging.info(f"Feature importance: {feature_importance}")
    functions.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None, flag_matrix='all',
                                 input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='xgb',
                                 flag_raw='')
    functions.correlation_matrix(
        X=X_train[functions.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
        flag_matrix='all',
        input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='xgb', flag_raw='yes')

    # Random Forest
    print(Fore.GREEN + '\n *** Starting Random Forest *** \n' + Style.RESET_ALL)
    logging.info('*** Starting Random Forest ***')
    X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, x_train_recall, x_train_f1 = functions.rand_forest(
        df=X_train, criterion=y_train, df_us=X_train_us,
        criterion_us=y_train_us,
        test_X_us=X_test_us,
        test_y_us=y_test_us,
        model_to_predict=None,
        predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features, cut_offs=cut_offs,
        params=params)
    X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, x_test_recall, x_test_f1  = functions.rand_forest(df=X_test,
                                                                                                 criterion=y_test,
                                                                                                 df_us=X_train_us,
                                                                                                 criterion_us=y_train_us,
                                                                                                 test_X_us=X_test_us,
                                                                                                 test_y_us=y_test_us,
                                                                                                 model_to_predict=model_train,
                                                                                                 predict_only_flag='yes',
                                                                                                 test_X=None,
                                                                                                 test_y=None,
                                                                                                 final_features=final_features,
                                                                                                 cut_offs=cut_offs,
                                                                                                 params=params)
    t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, t1df_recall, t1df_f1 = functions.rand_forest(df=t1df,
                                                                                       criterion=t1df[criterion_column],
                                                                                       df_us=X_train_us,
                                                                                       criterion_us=y_train_us,
                                                                                       test_X_us=X_test_us,
                                                                                       test_y_us=y_test_us,
                                                                                       model_to_predict=model_train,
                                                                                       predict_only_flag='yes',
                                                                                       test_X=None,
                                                                                       test_y=None,
                                                                                       final_features=final_features,
                                                                                       cut_offs=cut_offs, params=params)
    t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, t2df_recall, t2df_f1 = functions.rand_forest(df=t2df,
                                                                                       criterion=t2df[criterion_column],
                                                                                       df_us=X_train_us,
                                                                                       criterion_us=y_train_us,
                                                                                       test_X_us=X_test_us,
                                                                                       test_y_us=y_test_us,
                                                                                       model_to_predict=model_train,
                                                                                       predict_only_flag='yes',
                                                                                       test_X=None,
                                                                                       test_y=None,
                                                                                       final_features=final_features,
                                                                                       cut_offs=cut_offs, params=params)
    t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, t3df_recall, t3df_f1 = functions.rand_forest(df=t3df,
                                                                                       criterion=t3df[criterion_column],
                                                                                       df_us=X_train_us,
                                                                                       criterion_us=y_train_us,
                                                                                       test_X_us=X_test_us,
                                                                                       test_y_us=y_test_us,
                                                                                       model_to_predict=model_train,
                                                                                       predict_only_flag='yes',
                                                                                       test_X=None,
                                                                                       test_y=None,
                                                                                       final_features=final_features,
                                                                                       cut_offs=cut_offs, params=params)

    models = models.append(
        {'Method': 'rf', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec,
         'Recall': x_train_recall, 'F1': x_train_f1,
         'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'rf', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec,
         'Recall': x_test_recall, 'F1': x_test_f1,
         'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'rf', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec, 'Recall': t1df_recall,
         'F1': t1df_f1,
         'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'rf', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec, 'Recall': t2df_recall,
         'F1': t2df_f1,
         'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'rf', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec, 'Recall': t3df_recall,
         'F1': t3df_f1,
         'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
        ignore_index=True)
    pickle.dump(model_train, open(session_id_folder + '/rf/model_train.pkl', 'wb'))
    feature_importance.to_csv(session_id_folder + '/rf/feat_importance.csv', index=False)
    logging.info(f"Feature importance: {feature_importance}")
    print(f"\t Feature importance: {feature_importance}")
    functions.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None, flag_matrix='all',
                                 input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='rf',
                                 flag_raw='')
    functions.correlation_matrix(
        X=X_train[functions.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
        flag_matrix='all',
        input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='rf', flag_raw='yes')

    print(Fore.GREEN + '\n *** Starting Logistic regression *** \n' + Style.RESET_ALL)
    logging.info('*** Starting Logistic regression ***')
    try:
        X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, lr_table, x_train_recall, x_train_f1 = functions.lr_run(
            df=X_train, criterion=y_train,
            model_to_predict=None,
            predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features,
            lr_features=params, cut_offs=cut_offs, params=params)
        X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, _, x_test_recall, x_test_f1 = functions.lr_run(df=X_test,
                                                                                                   criterion=y_test,
                                                                                                   model_to_predict=model_train,
                                                                                                   predict_only_flag='yes',
                                                                                                   test_X=None,
                                                                                                   test_y=None,
                                                                                                   final_features=final_features,
                                                                                                   lr_features=None,
                                                                                                   cut_offs=cut_offs,
                                                                                                   params=params)
        t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, _, t1df_recall, t1df_f1 = functions.lr_run(df=t1df,
                                                                                         criterion=t1df[
                                                                                             criterion_column],
                                                                                         model_to_predict=model_train,
                                                                                         predict_only_flag='yes',
                                                                                         test_X=None,
                                                                                         test_y=None,
                                                                                         final_features=final_features,
                                                                                         lr_features=None,
                                                                                         cut_offs=cut_offs,
                                                                                         params=params)
        t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, _, t2df_recall, t2df_f1 = functions.lr_run(df=t2df,
                                                                                         criterion=t2df[
                                                                                             criterion_column],
                                                                                         model_to_predict=model_train,
                                                                                         predict_only_flag='yes',
                                                                                         test_X=None,
                                                                                         test_y=None,
                                                                                         final_features=final_features,
                                                                                         lr_features=None,
                                                                                         cut_offs=cut_offs,
                                                                                         params=params)
        t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, _, t3df_recall, t3df_f1 = functions.lr_run(df=t3df,
                                                                                         criterion=t3df[
                                                                                             criterion_column],
                                                                                         model_to_predict=model_train,
                                                                                         predict_only_flag='yes',
                                                                                         test_X=None,
                                                                                         test_y=None,
                                                                                         final_features=final_features,
                                                                                         lr_features=None,
                                                                                         cut_offs=cut_offs,
                                                                                         params=params)

        models = models.append(
            {'Method': 'lr', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec,
             'Recall': x_train_recall, 'F1': x_train_f1,
             'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'lr', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec,
             'Recall': x_test_recall, 'F1': x_test_f1,
             'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'lr', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec,
             'Recall': t1df_recall, 'F1': t1df_f1,
             'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'lr', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec,
             'Recall': t2df_recall, 'F1': t2df_f1,
             'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
            ignore_index=True)
        models = models.append(
            {'Method': 'lr', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec,
             'Recall': t3df_recall, 'F1': t3df_f1,
             'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
            ignore_index=True)
        pickle.dump(model_train, open(session_id_folder + '/lr/model_train.pkl', 'wb'))
        feature_importance.to_csv(session_id_folder + '/lr/feat_importance.csv', index=False)
        logging.info(f"Feature importance: {feature_importance}")
        print(f"\t Feature importance: {feature_importance}")
        lr_table.to_csv(session_id_folder + '/lr/lr_table.csv', index=False)
        functions.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None,
                                     flag_matrix='all',
                                     input_data_project_folder=None, session_id_folder=session_id_folder,
                                     model_corr='lr',
                                     flag_raw='')
        functions.correlation_matrix(
            X=X_train[functions.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
            flag_matrix='all',
            input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='lr', flag_raw='yes')
    except Exception as e:
        logging.error('LOGIT: %s', e)
        print('LOGIT error', e)
        pass

    print(Fore.GREEN + '\n *** Starting Decision Tree *** \n' + Style.RESET_ALL)
    logging.info('*** Starting Decision Tree ***')
    X_train, model_train, x_train_ac, x_train_auc, x_train_prec, x_train_nb_features, feature_importance, x_train_recall, x_train_f1 = functions.decision_tree(
        df=X_train, criterion=y_train,
        df_us=X_train_us,
        criterion_us=y_train_us,
        test_X_us=X_test_us,
        test_y_us=y_test_us,
        model_to_predict=None,
        predict_only_flag='no', test_X=X_test, test_y=y_test, final_features=final_features, cut_offs=cut_offs,
        params=params)
    X_test, _, x_test_ac, x_test_auc, x_test_prec, x_test_nb_features, _, x_test_recall, x_test_f1 = functions.decision_tree(df=X_test,
                                                                                                   criterion=y_test,
                                                                                                   df_us=X_train_us,
                                                                                                   criterion_us=y_train_us,
                                                                                                   test_X_us=X_test_us,
                                                                                                   test_y_us=y_test_us,
                                                                                                   model_to_predict=model_train,
                                                                                                   predict_only_flag='yes',
                                                                                                   test_X=None,
                                                                                                   test_y=None,
                                                                                                   final_features=final_features,
                                                                                                   cut_offs=cut_offs,
                                                                                                   params=params)
    t1df, _, t1df_ac, t1df_auc, t1df_prec, t1df_nb_features, _, t1df_recall, t1df_f1 = functions.decision_tree(df=t1df,
                                                                                         criterion=t1df[
                                                                                             criterion_column],
                                                                                         df_us=X_train_us,
                                                                                         criterion_us=y_train_us,
                                                                                         test_X_us=X_test_us,
                                                                                         test_y_us=y_test_us,
                                                                                         model_to_predict=model_train,
                                                                                         predict_only_flag='yes',
                                                                                         test_X=None,
                                                                                         test_y=None,
                                                                                         final_features=final_features,
                                                                                         cut_offs=cut_offs,
                                                                                         params=params)
    t2df, _, t2df_ac, t2df_auc, t2df_prec, t2df_nb_features, _, t2df_recall, t2df_f1 = functions.decision_tree(df=t2df,
                                                                                         criterion=t2df[
                                                                                             criterion_column],
                                                                                         df_us=X_train_us,
                                                                                         criterion_us=y_train_us,
                                                                                         test_X_us=X_test_us,
                                                                                         test_y_us=y_test_us,
                                                                                         model_to_predict=model_train,
                                                                                         predict_only_flag='yes',
                                                                                         test_X=None,
                                                                                         test_y=None,
                                                                                         final_features=final_features,
                                                                                         cut_offs=cut_offs,
                                                                                         params=params)
    t3df, _, t3df_ac, t3df_auc, t3df_prec, t3df_nb_features, _, t3df_recall, t3df_f1 = functions.decision_tree(df=t3df,
                                                                                         criterion=t3df[
                                                                                             criterion_column],
                                                                                         df_us=X_train_us,
                                                                                         criterion_us=y_train_us,
                                                                                         test_X_us=X_test_us,
                                                                                         test_y_us=y_test_us,
                                                                                         model_to_predict=model_train,
                                                                                         predict_only_flag='yes',
                                                                                         test_X=None,
                                                                                         test_y=None,
                                                                                         final_features=final_features,
                                                                                         cut_offs=cut_offs,
                                                                                         params=params)

    models = models.append(
        {'Method': 'dt', 'AccuracyScore': x_train_ac, 'AUC': x_train_auc, 'PrecisionScore': x_train_prec,
         'Recall': x_train_recall, 'F1': x_train_f1,
         'DataSet': 'X_train', 'NbFeatures': x_train_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'dt', 'AccuracyScore': x_test_ac, 'AUC': x_test_auc, 'PrecisionScore': x_test_prec,
         'Recall': x_test_recall, 'F1': x_test_f1,
         'DataSet': 'X_test', 'NbFeatures': x_test_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'dt', 'AccuracyScore': t1df_ac, 'AUC': t1df_auc, 'PrecisionScore': t1df_prec, 'Recall': t1df_recall,
         'F1': t1df_f1,
         'DataSet': 't1df', 'NbFeatures': t1df_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'dt', 'AccuracyScore': t2df_ac, 'AUC': t2df_auc, 'PrecisionScore': t2df_prec, 'Recall': t2df_recall,
         'F1': t2df_f1,
         'DataSet': 't2df', 'NbFeatures': t2df_nb_features},
        ignore_index=True)
    models = models.append(
        {'Method': 'dt', 'AccuracyScore': t3df_ac, 'AUC': t3df_auc, 'PrecisionScore': t3df_prec, 'Recall': t3df_recall,
         'F1': t3df_f1,
         'DataSet': 't3df', 'NbFeatures': t3df_nb_features},
        ignore_index=True)
    pickle.dump(model_train, open(session_id_folder + '/dt/model_train.pkl', 'wb'))
    feature_importance.to_csv(session_id_folder + '/dt/feat_importance.csv', index=False)
    logging.info(f"Feature importance: {feature_importance}")
    print(f"\t Feature importance: {feature_importance}")
    functions.correlation_matrix(X=X_train[feature_importance['columns'].unique().tolist()], y=None, flag_matrix='all',
                                 input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='dt',
                                 flag_raw='')
    functions.correlation_matrix(
        X=X_train[functions.raw_features_to_list(feature_importance['columns'].unique().tolist())], y=None,
        flag_matrix='all',
        input_data_project_folder=None, session_id_folder=session_id_folder, model_corr='dt', flag_raw='yes')

    # Create session folder and save results
    print(Fore.GREEN + '\n Saving Train, Test, T1,2,3 datasets (full datasets)' + Style.RESET_ALL)
    logging.info('\n Saving Train, Test, T1,2,3 datasets (full datasets)')

    X_train.to_parquet(session_id_folder + '/df_x_train.parquet')
    X_test.to_parquet(session_id_folder + '/df_x_test.parquet')
    t1df.to_parquet(session_id_folder + '/df_t1df.parquet')
    t2df.to_parquet(session_id_folder + '/df_t2df.parquet')
    t3df.to_parquet(session_id_folder + '/df_t3df.parquet')
    models.to_csv(session_id_folder + '/models.csv', index=False)
    print(Fore.YELLOW + '\n Session folder: ' + session_id_folder + Style.RESET_ALL)
    logging.info(f'Create session folder: {session_id_folder}')


if run_info == 'create':
    print('Creating the folder structure')
    logging.info('Creating the folder structure')
    if not os.path.isdir(log_folder_name):
        os.mkdir(log_folder_name)
    if not os.path.isdir(session_folder_name):
        os.mkdir(session_folder_name)
    if not os.path.isdir(input_data_folder_name):
        os.mkdir(input_data_folder_name)
    if not os.path.isdir(output_data_folder_name):
        os.mkdir(output_data_folder_name)
    if not os.path.isdir(functions_folder_name):
        os.mkdir(functions_folder_name)
    if not os.path.isdir(params_folder_name):
        os.mkdir(params_folder_name)
    sys.exit()
elif run_info == 'load':
    functions.print_load()
    print(Fore.GREEN + 'Starting the session for: ' + input_data_project_folder + Style.RESET_ALL)
    logging.info(f'Starting the session for: {input_data_project_folder}')
    input_df, input_df_full = functions.data_load(input_data_folder_name=input_data_folder_name,
                                                  input_data_project_folder=input_data_project_folder,
                                                  criterion_column=criterion_column,
                                                  periods_to_exclude=periods_to_exclude,
                                                  observation_date_column=observation_date_column, params=params)
    functions.data_cleaning(input_df=input_df, input_df_full=input_df_full,
                            output_data_folder_name=output_data_folder_name,
                            input_data_project_folder=input_data_project_folder,
                            columns_to_exclude=columns_to_exclude, criterion_column=criterion_column,
                            observation_date_column=observation_date_column, missing_treatment=missing_treatment,
                            params=params)
    functions.print_end()
elif run_info == 'train':
    functions.print_train()
    print(Fore.GREEN + 'Starting the session for: ' + input_data_project_folder + Style.RESET_ALL)
    logging.info(f'Starting the session for: {input_data_project_folder}')
    modeller(input_data_project_folder)
    functions.print_end()
elif run_info == 'eval':
    functions.print_eval()
    print(Fore.GREEN + 'Starting the session for: ' + input_data_project_folder + Style.RESET_ALL)
    logging.info(f'Starting the session for: {input_data_project_folder}')

    # Create new session folder:
    print(Fore.GREEN + 'Createing session folder. Starting' + Style.RESET_ALL)
    logging.info('Createing session folder. Starting')
    session_id = 'EVAL_' + input_data_project_folder + '_' + str(start) + '_' + tag
    session_id_folder = session_folder_name + session_id
    os.mkdir(session_id_folder)
    print('Createing session folder. Done')
    logging.info('Createing session folder. Done')

    try:
        print(Fore.GREEN + '\n Starting Eval for LR\n' + Style.RESET_ALL)
        functions.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                             session_id_folder, criterion_column,
                             observation_date_column,
                             columns_to_exclude,
                             periods_to_exclude,
                             t1df_period,
                             t2df_period,
                             t3df_period,
                             model_arg='lr',
                             missing_treatment=missing_treatment, params=params)
    except Exception as e:
        print(f'ERROR with LR: {e}')
        pass
    try:
        print(Fore.GREEN + '\n Starting Eval for DT\n'+Style.RESET_ALL)
        functions.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                             session_id_folder, criterion_column,
                             observation_date_column,
                             columns_to_exclude,
                             periods_to_exclude,
                             t1df_period,
                             t2df_period,
                             t3df_period,
                             model_arg='dt',
                             missing_treatment=missing_treatment, params=params)
    except Exception as e:
        print(f'ERROR with DT: {e}')
        pass
    try:
        print(Fore.GREEN + '\n Starting Eval for XGB \n' + Style.RESET_ALL)
        functions.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                             session_id_folder, criterion_column,
                             observation_date_column,
                             columns_to_exclude,
                             periods_to_exclude,
                             t1df_period,
                             t2df_period,
                             t3df_period,
                             model_arg='xgb',
                             missing_treatment=missing_treatment, params=params)
    except Exception as e:
        print(f'ERROR with XGB: {e}')
        pass
    try:
        print(Fore.GREEN + '\n Starting Eval for RF\n' + Style.RESET_ALL)
        functions.merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
                             session_id_folder, criterion_column,
                             observation_date_column,
                             columns_to_exclude,
                             periods_to_exclude,
                             t1df_period,
                             t2df_period,
                             t3df_period,
                             model_arg='rf',
                             missing_treatment=missing_treatment, params=params)
    except Exception as e:
        print(f'ERROR with RF: {e}')
        pass

    functions.print_end()
else:
    print('No arguments')
    functions.print_end()

sys.exit()
