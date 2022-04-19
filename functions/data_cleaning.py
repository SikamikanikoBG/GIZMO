import logging
import sys
from os import listdir
from os.path import join, isfile

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from optbinning import OptimalBinning
from sklearn.model_selection import train_test_split

import functions


def data_load(session):
    print('\n Starting data load... \n')
    logging.info('\n Data load...')
    onlyfiles = [f for f in listdir(session.input_data_folder_name + session.input_data_project_folder + '/') if
                 isfile(join(session.input_data_folder_name + session.input_data_project_folder + '/', f))]
    if len(onlyfiles) == 0:
        logging.error('ERROR: No files in input folder. Aborting the program.')
        sys.exit()

    if 'dict' in onlyfiles[0]:
        input_file = onlyfiles[1]
    else:
        input_file = onlyfiles[0]

    _, _, extention = str(input_file).partition('.')

    if 'csv' not in extention:
        logging.error('ERROR: input data not a csv file.')
        sys.exit()

    input_df = pd.read_csv(session.input_data_folder_name + session.input_data_project_folder + '/' + input_file)
    print(f"Loading file {input_file}")
    logging.info(f"\n Loading file {input_file}")
    if len(input_df.columns.to_list()) == 1:
        input_df = pd.read_csv(session.input_data_folder_name + session.input_data_project_folder + '/' + input_file,
                               sep=';')
        if len(input_df.columns.to_list()) == 1:
            logging.error('ERROR: input data separator not any of the following ,;')
            sys.exit()

    # Check if multiclass classification criterion is passed
    if input_df[session.params['criterion_column']].nunique() > 2:
        print()
        logging.error('ERROR: Multiclass classification criterion passed')
        sys.exit()

    # Remove periods:
    logging.info('LOAD: DataFrame len: %s, periods to exclude: %s',
                 len(input_df), session.params['periods_to_exclude'])
    for el in session.params['periods_to_exclude']:
        input_df = input_df[input_df[session.params['observation_date_column']] != el]
    logging.info('LOAD:DataFrame len: %s, periods kept after exclusion: %s',
                 len(input_df), input_df[session.params['observation_date_column']].unique().tolist())

    input_df_full = pd.DataFrame()
    if session.params["under_sampling"]:
        criterion_rate = round(
            input_df[session.params['criterion_column']].sum() / input_df[session.params['criterion_column']].count(),
            2)
        input_df_full = input_df.copy()
        print(
            f'Starting undersampling with strategy: {session.params["under_sampling"]}. The initial dataframe lenght is {input_df.shape} and criterion rate: {criterion_rate}')
        logging.info(
            f'\n Starting undersampling with strategy: {session.params["under_sampling"]}. The initial dataframe lenght is {input_df.shape} and criterion rate: {criterion_rate}')
        under = RandomUnderSampler(
            sampling_strategy=session.params["under_sampling"])  # define strategy for undersampling
        under_x, under_y = under.fit_resample(input_df, input_df[
            session.params['criterion_column']])  # create new df (under_X)  undersampled based on above strategy
        input_df = under_x
        criterion_rate = round(
            input_df[session.params['criterion_column']].sum() / input_df[session.params['criterion_column']].count(),
            2)
        print(
            f'Undersampling done. The new dataframe lenght is {input_df.shape} and criterion rate: {criterion_rate}')
        logging.info(
            f'Undersampling done. The new dataframe lenght is {input_df.shape} and criterion rate: {criterion_rate}')

    return input_df, input_df_full


def data_cleaning(input_df, input_df_full, session):  # start data cleaning
    print('\n Starting data clean... \n')
    logging.info('\n Data clean...')

    # input_df = outliar(df=input_df)

    print('\n Splitting columns by types \n')
    logging.info('\n Splitting columns by types \n')
    categorical_cols, numerical_cols, object_cols, dates_cols = split_columns_by_types(df=input_df)
    for el in categorical_cols[:]:
        if el in session.params['columns_to_exclude']:
            categorical_cols.remove(el)

    for el in numerical_cols[:]:
        if el in session.params['columns_to_exclude']:
            numerical_cols.remove(el)

    for el in object_cols[:]:
        if el in session.params['columns_to_exclude']:
            object_cols.remove(el)

    for el in dates_cols[:]:
        if el in session.params['columns_to_exclude']:
            dates_cols.remove(el)

    # treat specified numerical as objects
    print('\n Switching type from number to object since nb of categories is below 20 \n')
    logging.info('\n Switching type from number to object since nb of categories is below 20 \n')
    for el in numerical_cols[:]:
        if len(input_df[el].unique()) < 20:
            numerical_cols.remove(el)
            object_cols.append(el)
            print(
                '\t Switching type for {} from number to object since nb of categories is below 20 ({})'.format(el,
                                                                                                                len(
                                                                                                                    input_df[
                                                                                                                        el].unique())))
            logging.info(
                'Switching type for {} from number to object since nb of categories is below 20 ({})'.format(el,
                                                                                                             len(
                                                                                                                 input_df[
                                                                                                                     el].unique())))

    # convert objects to categories and get dummies
    print('\n Converting objects to dummies \n')
    logging.info('\n Converting objects to dummies \n')

    for col in object_cols:
        dummies = pd.get_dummies(input_df[col], prefix=col + '_dummie')
        input_df[dummies.columns] = dummies
        if session.params["under_sampling"]:
            dummies = pd.get_dummies(input_df_full[col], prefix=col + '_dummie')
            input_df_full[dummies.columns] = dummies

    object_cols_cat = []
    for col in input_df.columns.to_list():
        if '_cat' in col:
            object_cols_cat.append(col)

    object_cols_dummie = []
    for col in input_df.columns.to_list():
        if '_dummie' in col:
            object_cols_dummie.append(col)

    final_features = object_cols_dummie + object_cols_cat + numerical_cols

    # Check correlation. # remove low correlation columns from numerical. At this point this is needed to lower the nb of ratios to be created
    print('\n Removing low correlation columns from numerical \n')
    logging.info('\n Removing low correlation columns from numerical \n')
    final_features = functions.correlation_matrix(X=input_df[final_features],
                                                  y=input_df[session.params['criterion_column']],
                                                  input_data_project_folder=session.input_data_project_folder,
                                                  flag_matrix=None,
                                                  session_id_folder=None, model_corr='', flag_raw='')

    for el in numerical_cols[:]:
        if el in final_features:
            pass
        else:
            numerical_cols.remove(el)
            # print(f"\t {el} removed due to low correlation vs the criterion")
            logging.info(f"{el} removed due to low correlation vs the criterion")

    # Feature engineering
    print('\n Creating ratios with numerical columns \n')
    logging.info('\n Creating ratios with numerical columns \n')
    input_df = functions.create_ratios(df=input_df, columns=numerical_cols)
    if session.params["under_sampling"]:
        input_df_full = functions.create_ratios(df=input_df_full, columns=numerical_cols)

    ratios_cols = []
    for col in input_df.columns.to_list():
        if '_ratio_' in col:
            ratios_cols.append(col)

    final_features = object_cols_dummie + object_cols_cat + numerical_cols + ratios_cols

    # Check correlation
    print('\n Removing low correlation columns from ratios \n')
    logging.info('\n Removing low correlation columns from ratios \n')
    final_features = functions.correlation_matrix(X=input_df[final_features],
                                                  y=input_df[session.params['criterion_column']],
                                                  input_data_project_folder=session.input_data_project_folder,
                                                  flag_matrix=None,
                                                  session_id_folder=None, model_corr='', flag_raw='')

    for el in ratios_cols[:]:
        if el in final_features:
            pass
        else:
            ratios_cols.remove(el)
            print(f"\t {el} removed due to low correlation vs the criterion")
            logging.info(f"{el} removed due to low correlation vs the criterion")

    binned_numerical = numerical_cols + ratios_cols
    # Optimal binning
    input_df, input_df_full, binned_numerical = optimal_binning(df=input_df, df_full=input_df_full,
                                                                columns=binned_numerical,
                                                                criterion_column=session.params['criterion_column'],
                                                                final_features=final_features,
                                                                observation_date_column=session.params[
                                                                    'observation_date_column'], params=session.params)
    final_features = final_features + binned_numerical
    final_features = list(set(final_features))
    for el in final_features[:]:
        if "dummie" not in el:
            final_features.remove(el)
        else:
            pass
    logging.info(f'Final features: {final_features}')

    # Check correlation
    print('\n Checking correlation one more time now with binned \n')
    logging.info('\n Checking correlation one more time now with binned \n')
    final_features = functions.correlation_matrix(X=input_df[final_features],
                                                  y=input_df[session.params['criterion_column']],
                                                  input_data_project_folder=session.input_data_project_folder,
                                                  flag_matrix=None,
                                                  session_id_folder=None, model_corr='', flag_raw='')

    # all_columns = final_features + [session.params['criterion_column']] + [
    #     session.params['observation_date_column']] + object_cols
    # if session.params['secondary_criterion_columns']:
    #     for el in session.params['secondary_criterion_columns']:
    #         all_columns.append(el)
    #     all_columns = list(dict.fromkeys(all_columns[:]))

    # Finalizing the dataframes
    input_df = missing_values(df=input_df, missing_treatment=session.params['missing_treatment'],
                              input_data_project_folder=session.input_data_project_folder)
    if session.params['under_sampling']:
        input_df_full = missing_values(df=input_df_full, missing_treatment=session.params['missing_treatment'],
                                       input_data_project_folder=session.input_data_project_folder)
    all_columns = input_df.columns.to_list()

    return input_df[all_columns], input_df_full[all_columns], final_features


def missing_values(df, missing_treatment, input_data_project_folder):
    # logging

    # removing columns with 50% or more missing values
    print('\n Treating missing values as specified in the param file \n')
    logging.info('\n Treating missing values as specified in the param file \n')
    missing_cols = df[df.columns[df.isnull().mean() > 0.99]].columns.to_list()
    logging.info('MISSING: the following columns are with >50 perc of missing values and will be deleted: %s',
                 missing_cols)
    percent_missing = df.isna().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df.to_csv('./output_data/' + input_data_project_folder + '/missing_values.csv', index=False)
    df = df.drop(columns=missing_cols)

    # drop rows with na
    len_before = len(df)

    if missing_treatment == 'delete':
        df = df.dropna()
    elif missing_treatment == 'column_mean':
        column_means = df.mean()
        df = df.fillna(column_means)
        df = df.dropna()
    elif missing_treatment == 'median':
        column_median = df.median()
        df = df.fillna(column_median)
        df = df.dropna()
    else:
        df = df.fillna(missing_treatment)
        if df.isnull().values.any():
            print(
                '\t WARNING: Not all missing were treated! The following missing in the columns will be filled with: 0 for numerical, 1900-01-01 for dates, True for bool, else MissingInformation string')

            for f in df.columns:

                # integer
                if df[f].dtype == "int":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    logging.info(f'{f} filled with 0')

                elif df[f].dtype == "float":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    logging.info(f'{f} filled with 0')

                elif df[f].dtype == "uint":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    logging.info(f'{f} filled with 0')

                # dates
                elif df[f].dtype == '<M8[ns]':
                    df[f] = df[f].fillna(pd.to_datetime('1900-01-01'))
                    # print(f'{f} filled with 1900-01-01')
                    logging.info(f'{f} filled with 1900-01-01')

                # boolean
                elif df[f].dtype == 'bool':
                    df[f] = df[f].fillna(True)
                    # print(f'{f} filled with True')
                    logging.info(f'{f} filled with True')

                # string
                else:
                    df[f] = df[f].fillna('MissingInformation')
                    # print(f'{f} filled with MissingInformation')
                    logging.info(f'{f} filled with MissingInformation')

    len_after = len(df)
    removed_missing_rows = len_before - len_after
    logging.info('MISSING: rows removed due to missing: %s (%s)', removed_missing_rows,
                 round(removed_missing_rows / len_before, 2))

    return df


def outliar(df):
    return df


def split_columns_by_types(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = df.select_dtypes(include=numerics).columns.to_list()
    object_cols = df.select_dtypes(include=['object']).columns.to_list()

    print('\t Removing from the list for having too many categories')
    for col in object_cols[:]:
        if len(df[col].unique()) > 20:
            print('\t Removing {} from the list for having too many categories ({})'.format(col, len(df[col].unique())))
            logging.info(
                'Removing {} from the list for having too many categories ({})'.format(col, len(df[col].unique())))
            object_cols.remove(col)

    dates_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.to_list()
    categorical_cols = df.select_dtypes(include=['category']).columns.to_list()

    logging.info('Columns splitted by types cat: %s', categorical_cols)
    logging.info('Columns splitted by types num: %s', numerical_cols)
    logging.info('Columns splitted by types obj: %s', object_cols)
    logging.info('Columns splitted by types dat: %s', dates_cols)

    return categorical_cols, numerical_cols, object_cols, dates_cols


def convert_category(df, column):
    df[column] = df[column].astype('category')
    df[column + "_cat"] = df[column].cat.codes
    return df


def optimal_binning(df, df_full, columns, criterion_column, final_features, observation_date_column, params):
    # creating binned dummie features from all numeric ones
    print('\n Starting numerical features Optimal binning (max 4 bins) based on train df \n')

    for col in columns[:]:
        temp_df = df.copy()
        temp_df2 = df.copy()
        temp_df_full = df_full.copy()

        # Removing all periods before splitting train and test
        temp_df2 = temp_df2[temp_df2[observation_date_column] != params['t1df']]
        temp_df2 = temp_df2[temp_df2[observation_date_column] != params['t2df']]
        temp_df2 = temp_df2[temp_df2[observation_date_column] != params['t3df']]

        x_train, _, y_train, _ = train_test_split(
            temp_df2, temp_df2[criterion_column], test_size=0.33, random_state=42)
        x_train = x_train.dropna(subset=[col])

        x = x_train[col].values
        y = x_train[criterion_column].values
        optb = OptimalBinning(name=col, dtype='numerical', solver='cp', max_n_bins=3, min_bin_size=0.1)
        optb.fit(x, y)

        temp_df = temp_df.dropna(subset=[col])
        binned_col_name = col + '_binned'
        temp_df[binned_col_name] = optb.transform(temp_df[col], metric='bins')

        dummies = pd.get_dummies(temp_df[binned_col_name], prefix=binned_col_name + '_dummie')
        print('\t {} is with the following splits: {} and dummie columns: {}'.format(col, optb.splits,
                                                                                     list(dummies.columns)))
        logging.info('{} is with the following splits: {} and dummie columns: {}'.format(col, optb.splits,
                                                                                         list(dummies.columns)))
        temp_df[dummies.columns] = dummies

        columns.remove(col)
        for el in list(dummies.columns):
            columns.append(el)

        df = pd.concat([df, temp_df[dummies.columns]], axis=1)
        df[dummies.columns] = df[dummies.columns].fillna(0)

        if params['under_sampling']:
            temp_df_full = temp_df_full.dropna(subset=[col])
            binned_col_name = col + '_binned'
            temp_df_full[binned_col_name] = optb.transform(temp_df_full[col], metric='bins')

            dummies = pd.get_dummies(temp_df_full[binned_col_name], prefix=binned_col_name + '_dummie')
            temp_df_full[dummies.columns] = dummies

            df_full = pd.concat([df_full, temp_df_full[dummies.columns]], axis=1)
            df_full[dummies.columns] = df_full[dummies.columns].fillna(0)

    # Recoding strings
    for string in columns[:]:
        new_string = string.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        columns.remove(string)
        columns.append(new_string)

    for string in final_features[:]:
        new_string = string.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        final_features.remove(string)
        final_features.append(new_string)

    for col in df:
        new_string = col.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        df.rename(columns={col: new_string}, inplace=True)

    for col in df_full:
        new_string = col.replace("<", "less_than")
        new_string = new_string.replace(">", "more_than")
        new_string = new_string.replace(",", "_to_")
        new_string = new_string.replace("[", "from_incl_")
        new_string = new_string.replace("]", "_incl_")
        new_string = new_string.replace("(", "from_excl_")
        new_string = new_string.replace(")", "_excl_")
        new_string = new_string.replace(" ", "")
        df_full.rename(columns={col: new_string}, inplace=True)
    return df, df_full, columns
