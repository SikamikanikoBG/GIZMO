import sys

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from optbinning import OptimalBinning
from sklearn.model_selection import train_test_split

from src.functions.modeller import cut_into_bands
from src.functions.printing_and_logging_functions import print_and_log


def remove_column_if_not_in_final_features(final_features, numerical_cols):
    for el in numerical_cols[:]:
        if el in final_features:
            pass
        else:
            numerical_cols.remove(el)
            print_and_log(f"{el} removed due to low correlation vs the criterion", 'YELLOW')


def convert_obj_to_cat_and_get_dummies(input_df, input_df_full, object_cols, params):
    for col in object_cols:
        dummies = pd.get_dummies(input_df[col], prefix=col + '_dummie')
        input_df[dummies.columns] = dummies
        if params["under_sampling"]:
            dummies = pd.get_dummies(input_df_full[col], prefix=col + '_dummie')
            input_df_full[dummies.columns] = dummies


def switch_numerical_to_object_column(input_df, numerical_cols, object_cols):
    for el in numerical_cols[:]:
        if len(input_df[el].unique()) < 20:
            numerical_cols.remove(el)
            object_cols.append(el)
            print_and_log(
                'Switching type for {} from number to object '
                'since nb of categories is below 20 ({})'.format(el, len(input_df[el].unique())), '')


def remove_columns_to_exclude(categorical_cols, dates_cols, numerical_cols, object_cols, params):
    """
    Removes columns, specified in the param file
    Args:
        categorical_cols:
        dates_cols:
        numerical_cols:
        object_cols:
        params:

    Returns: nothing

    """
    for el in categorical_cols[:]:
        if el in params['columns_to_exclude']:
            categorical_cols.remove(el)
    for el in numerical_cols[:]:
        if el in params['columns_to_exclude']:
            numerical_cols.remove(el)
    for el in object_cols[:]:
        if el in params['columns_to_exclude']:
            object_cols.remove(el)
    for el in dates_cols[:]:
        if el in params['columns_to_exclude']:
            dates_cols.remove(el)


def missing_values(df, missing_treatment, input_data_project_folder):
    # logging

    # removing columns with 50% or more missing values
    print_and_log('\n Treating missing values as specified in the param file \n', 'GREEN')
    missing_cols = df[df.columns[df.isnull().mean() > 0.99]].columns.to_list()
    print_and_log(f'MISSING: the following columns are with >50 perc of missing values '
                  f'and will be deleted: {missing_cols}', '')
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
            print_and_log('\t WARNING: Not all missing were treated! The following missing in the columns will be '
                          'filled with: 0 for numerical, 1900-01-01 for dates, True for bool, else MissingInformation string',
                          'YELLOW')

            for f in df.columns:

                # integer
                if df[f].dtype == "int":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    print_and_log(f'{f} filled with 0', '')

                elif df[f].dtype == "float":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    print_and_log(f'{f} filled with 0', '')

                elif df[f].dtype == "uint":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    print_and_log(f'{f} filled with 0', '')

                # dates
                elif df[f].dtype == '<M8[ns]':
                    df[f] = df[f].fillna(pd.to_datetime('1900-01-01'))
                    # print(f'{f} filled with 1900-01-01')
                    print_and_log(f'{f} filled with 1900-01-01', '')

                # boolean
                elif df[f].dtype == 'bool':
                    df[f] = df[f].fillna(True)
                    # print(f'{f} filled with True')
                    print_and_log(f'{f} filled with True', '')

                # string
                else:
                    df[f] = df[f].fillna('MissingInformation')
                    # print(f'{f} filled with MissingInformation')
                    print_and_log(f'{f} filled with MissingInformation', '')

    len_after = len(df)
    removed_missing_rows = len_before - len_after
    print_and_log(f'MISSING: rows removed due to missing: '
                  f'{removed_missing_rows} ({round(removed_missing_rows / len_before, 2)}', '')

    return df


def outlier(df):
    return df


def split_columns_by_types(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = df.select_dtypes(include=numerics).columns.to_list()
    object_cols = df.select_dtypes(include=['object']).columns.to_list()

    print_and_log('\t Removing from the list for having too many categories', 'YELLOW')
    for col in object_cols[:]:
        if len(df[col].unique()) > 20:
            print_and_log('\t\tRemoving {} from the list for having '
                          'too many categories ({})'.format(col, len(df[col].unique())), 'YELLOW')
            object_cols.remove(col)

    dates_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.to_list()
    categorical_cols = df.select_dtypes(include=['category']).columns.to_list()

    print_and_log(f'Columns split by types cat: {categorical_cols}', '')
    print_and_log(f'Columns split by types num: {numerical_cols}', '')
    print_and_log(f'Columns split by types obj: {object_cols}', '')
    print_and_log(f'Columns split by types dat: {dates_cols}', '')

    return categorical_cols, numerical_cols, object_cols, dates_cols


def convert_column_to_category(df, column):
    df[column] = df[column].astype('category')
    df[column + "_cat"] = df[column].cat.codes
    return df


def optimal_binning(df, df_full, columns, criterion_column, final_features, observation_date_column, params):
    # creating binned dummie features from all numeric ones
    print_and_log('\n Starting numerical features Optimal binning (max 4 bins) based on train df \n', '')

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
        print_and_log('{} is with the following splits: {} and dummie columns: {}'.format(col, optb.splits,
                                                                                          list(dummies.columns)), '')
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


def check_if_multiclass_criterion_is_passed(input_df, params):
    if input_df[params['criterion_column']].nunique() > 2:
        print_and_log('ERROR: Multiclass classification criterion passed', 'RED')
        sys.exit()


def create_dict_based_on_col_name_contains(input_df_cols, text):
    collection_cols = []
    for col in input_df_cols:
        if text in col:
            collection_cols.append(col)
    return collection_cols


def create_ratios(df, columns):
    temp = []
    for col in columns:
        temp.append(col)
        for col2 in columns:
            if col2 in temp:
                pass
            else:
                df[col + '_div_ratio_' + col2] = df[col] / df[col2]
                df[col + '_div_ratio_' + col2] = df[col + '_div_ratio_' + col2].replace([np.inf, -np.inf], np.nan)
    print_and_log('Feat eng: Ratios created', '')
    return df


def create_tree_feats(df, columns, criterion):
    for col in columns:
        print(f"Trying tree feature for {col}")
        df[col + '_tree'], _ = cut_into_bands(X=df[[col]], y=df[criterion], depth=1)

        if df[col + '_tree'].nunique() == 1:
            del df[col + '_tree']
        else:
            class0_val = str(round(df[df[col + '_tree'] == 0][col].max(), 4))
            df = df.rename(columns={col + "_tree": col + "_tree_" + class0_val})
            print_and_log(
                'Feature engineering: New feature added with Decision Tree {}'.format(col + "_tree_" + class0_val), '')
    print_and_log('Feat eng: trees created', 'GREEN')
    return df


def correlation_matrix(X, y, input_data_project_folder, flag_matrix, session_id_folder, model_corr, flag_raw):
    corr_cols = []
    if flag_matrix != 'all':
        a = X.corrwith(y)

        a.to_csv('./output_data/' + input_data_project_folder + '/correl.csv')
        a = abs(a)
        b = a[a <= 0.05]
        c = a[a >= 0.4]

        a = a[a > 0.05]
        a = a[a < 0.4]

        corr_cols = a.index.to_list()
        corr_cols_removed = b.index.to_list()
        corr_cols_removed_c = c.index.to_list()

        for el in corr_cols_removed_c:
            if el in corr_cols_removed[:]:
                pass
            else:
                corr_cols_removed.append(el)
        print_and_log(f'Feat eng: keep only columns with correlation > 0.05: {corr_cols}', '')
    else:
        a = X.corr()
        if flag_raw == 'yes':
            a.to_csv(session_id_folder + '/' + model_corr + '/correl_raw_features.csv')
        else:
            a.to_csv(session_id_folder + '/' + model_corr + '/correl_features.csv')

    return corr_cols


def remove_periods_from_df(input_df, params):
    print_and_log(f'LOAD: DataFrame len: {len(input_df)}, periods to exclude: {params["periods_to_exclude"]}', '')
    for el in params['periods_to_exclude']:
        input_df = input_df[input_df[params['observation_date_column']] != el]
    print_and_log(f'LOAD: DataFrame len: {len(input_df)}, periods kept after '
                  f'exclusion: {input_df[params["observation_date_column"]].unique().tolist()}', '')
    return input_df


def under_sampling_df_based_on_params(input_df, params):
    """
    Under-sampling procedure for dataframe based on params
    Args:
        input_df:
        params:

    Returns: 2 dataframes - one with the under-sampled df and one with the original (full)

    """
    criterion_rate = round(
        input_df[params['criterion_column']].sum() / input_df[params['criterion_column']].count(),
        2)
    input_df_full = input_df.copy()
    print_and_log(f'\n Starting under-sampling with strategy: {params["under_sampling"]}. '
                  'The initial dataframe length is {input_df.shape} and criterion rate: {criterion_rate}', 'GREEN')

    # define strategy for under-sampling
    under = RandomUnderSampler(sampling_strategy=params["under_sampling"])
    # create new df (under_X)  under-sampled based on above strategy
    under_x, under_y = under.fit_resample(input_df, input_df[params['criterion_column']])
    input_df = under_x

    criterion_rate = round(
        input_df[params['criterion_column']].sum() / input_df[params['criterion_column']].count(),
        2)
    print_and_log(f'Under-sampling done. The new dataframe length is {input_df.shape} and '
                  'criterion rate: {criterion_rate}', 'GREEN')
    return input_df, input_df_full

def raw_features_to_list(final_features):
    raw_features = []
    for feat in final_features[:]:
        if 'binned' in feat:
            prefix, _, _ = str(feat).partition('_binned')
            if '_ratio_' in prefix:
                if '_div_' in prefix:
                    a, b, c = str(prefix).partition('_div_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_add_' in prefix:
                    a, b, c = str(prefix).partition('_add_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_subs_' in prefix:
                    a, b, c = str(prefix).partition('_subs_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
                elif '_mult_' in prefix:
                    a, b, c = str(prefix).partition('_mult_ratio_')
                    raw_features.append(a)
                    raw_features.append(c)
            else:
                raw_features.append(prefix)
        elif '_ratio_' in feat:
            if '_div_' in feat:
                a, b, c = str(feat).partition('_div_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_add_' in feat:
                a, b, c = str(feat).partition('_add_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_subs_' in feat:
                a, b, c = str(feat).partition('_subs_ratio_')
                raw_features.append(a)
                raw_features.append(c)
            elif '_mult_' in feat:
                a, b, c = str(feat).partition('_mult_ratio_')
                raw_features.append(a)
                raw_features.append(c)
        else:
            raw_features.append(feat)
    raw_features = list(dict.fromkeys(raw_features))
    return raw_features
