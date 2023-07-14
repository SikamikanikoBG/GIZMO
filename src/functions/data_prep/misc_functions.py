import sys

import numpy as np
import pandas as pd
from scipy import stats

import definitions
from src.functions.modelling.modelling_functions import cut_into_bands
from src.functions.printing_and_logging import print_and_log


def remove_column_if_not_in_final_features(final_features, numerical_cols, keep_cols):
    removed = []
    for el in numerical_cols[:]:
        if el in final_features:
            pass
        else:
            numerical_cols.remove(el)
            removed.append(el)

    for col in removed:
        for col2 in keep_cols:
            if col == col2:
                numerical_cols.append(col)
            elif col in col2:
                numerical_cols.append(col)

    if len(removed) > 30:
        print_and_log(f"[ REMOVE COLUMNS ] Removed due to low correlation vs the criterion: {len(removed)}", 'YELLOW')
    else:
        print_and_log(f"[ REMOVE COLUMNS ] Removed due to low correlation vs the criterion: {removed}", 'YELLOW')
    return final_features, numerical_cols


def convert_obj_to_cat_and_get_dummies(input_df, input_df_full, object_cols, params):
    for col in object_cols:
        dummies = pd.get_dummies(input_df[col], prefix=col + '_dummie', drop_first=True, dummy_na=True)
        input_df[dummies.columns] = dummies
        if params["under_sampling"]:
            dummies = pd.get_dummies(input_df_full[col], prefix=col + '_dummie', drop_first=True, dummy_na=True)
            input_df_full[dummies.columns] = dummies
    return input_df, input_df_full


def switch_numerical_to_object_column(input_df, numerical_cols, object_cols):
    switched = []
    for el in numerical_cols[:]:
        if len(input_df[el].unique()) < 20:
            numerical_cols.remove(el)
            object_cols.append(el)
            switched.append(el)
    """print_and_log(
                f'[ SWITCH TYPE ] Switching type from number to object since nb of categories is below 20 ({switched})', '')"""
    return numerical_cols, object_cols


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
    return categorical_cols, dates_cols, numerical_cols, object_cols


def outlier(df):
    return df


def split_columns_by_types(df, params):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = df.select_dtypes(include=numerics).columns.to_list()
    object_cols = df.select_dtypes(include=['object']).columns.to_list()
    dates_cols = df.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns.to_list()
    categorical_cols = df.select_dtypes(include=['category']).columns.to_list()

    """print_and_log(f'[ SPLIT COLUMNS ] Columns split by types cat: {len(categorical_cols)}', '')
    print_and_log(f'[ SPLIT COLUMNS ] Columns split by types num: {numerical_cols}', '')
    print_and_log(f'[ SPLIT COLUMNS ] Columns split by types obj: {object_cols}', '')
    print_and_log(f'[ SPLIT COLUMNS ] Columns split by types dat: {dates_cols}', '')"""

    categorical_cols, dates_cols, numerical_cols, object_cols = remove_columns_to_exclude(categorical_cols,
                                                                                          dates_cols,
                                                                                          numerical_cols,
                                                                                          object_cols,
                                                                                          params)

    return categorical_cols, numerical_cols, object_cols, dates_cols


def remove_categorical_cols_with_too_many_values(df, object_cols):
    print_and_log('[ REMOVE COLUMNS ] Removing from the list for having too many categories', 'YELLOW')
    for col in object_cols[:]:
        if len(df[col].unique()) > 20:
            print_and_log('[ REMOVE COLUMNS ] Removing {} from the list for having '
                          'too many categories ({})'.format(col, len(df[col].unique())), 'YELLOW')
            object_cols.remove(col)
    return object_cols


def convert_column_to_category(df, column):
    df[column] = df[column].astype('category')
    df[column + "_cat"] = df[column].cat.codes
    return df


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


def create_ratios(df, columns, columns_to_include):
    shortlist = []
    temp = []

    if columns_to_include:
        for col in columns:
            for col_keep in columns_to_include:
                if col == col_keep or col in col_keep:
                    if col not in shortlist:
                        shortlist.append(col)

        for col in shortlist:
            for col2 in shortlist:
                df[col + '_div_ratio_' + col2] = df[col] / df[col2]
                df[col + '_div_ratio_' + col2] = df[col + '_div_ratio_' + col2].replace([np.inf, -np.inf], np.nan)
    else:
        for col in columns:
            temp.append(col)
            for col2 in columns:
                if col2 in temp:
                    pass
                else:
                    df[col + '_div_ratio_' + col2] = df[col] / df[col2]
                    df[col + '_div_ratio_' + col2] = df[col + '_div_ratio_' + col2].replace([np.inf, -np.inf], np.nan)
    print_and_log(f'[ RATIOS ] Feat eng: Ratios created:', '')
    return df


def create_tree_feats(df, columns, criterion):
    for col in columns:
        print_and_log(f"[ TREES ] Trying tree feature for {col}", "")
        df[col + '_tree'], _ = cut_into_bands(X=df[[col]], y=df[criterion], depth=1)

        if df[col + '_tree'].nunique() == 1:
            del df[col + '_tree']
        else:
            class0_val = str(round(df[df[col + '_tree'] == 0][col].max(), 4))
            df = df.rename(columns={col + "_tree": col + "_tree_" + class0_val})
            print_and_log(
                '[ FEATURE ENGINEERING ] Feature engineering: New feature added with Decision Tree {}'.format(col + "_tree_" + class0_val), '')
    print_and_log('[ RATIOS ] Feat eng: trees created', 'GREEN')
    return df


def correlation_matrix(X, y, input_data_project_folder, flag_matrix, session_id_folder, model_corr, flag_raw, keep_cols):
    corr_cols = []
    if flag_matrix != 'all':
        a = X.corrwith(y)

        a.to_csv(definitions.ROOT_DIR  + '/output_data/' + input_data_project_folder + '/correl.csv')
        a = abs(a)
        b = a[a <= 0.05]
        c = a[a >= 0.4]

        a = a[a > 0.05]
        a = a[a < 0.4]

        if len(a) > 100:
            a = a.sort_values(ascending=False)
            a = a[:100]

        corr_cols = a.index.to_list()
        corr_cols_removed = b.index.to_list()
        corr_cols_removed_c = c.index.to_list()

        for el in corr_cols_removed_c:
            if el in corr_cols_removed[:]:
                pass
            else:
                corr_cols_removed.append(el)

        for col in corr_cols_removed[:]:
            for col2 in keep_cols:
                if col == col2:
                    corr_cols.append(col)
                elif col in col2:
                    corr_cols.append(col)

        print_and_log(f'[ CORRELATION ] Feat eng: keep only columns with correlation > 0.05: {len(corr_cols)} columns', '')
    else:
        a = X.corr()
        if flag_raw == 'yes':
            a.to_csv(session_id_folder + '/' + model_corr + '/correl_raw_features.csv')
        else:
            a.to_csv(session_id_folder + '/' + model_corr + '/correl_features.csv')

    return corr_cols


def remove_periods_from_df(input_df, params):
    print_and_log(f'[ PERIODS ] LOAD: DataFrame len: {len(input_df)}, periods to exclude: {params["periods_to_exclude"]}', '')
    for el in params['periods_to_exclude']:
        input_df = input_df[input_df[params['observation_date_column']] != el]
    return input_df


def treating_outliers(input_df, secondary_input_df):
    thres = 3 # threshold for zscore outliers

    for col in input_df.columns:
        input_df["zscore"] = stats.zscore(input_df[col])
        if len(secondary_input_df) > 0:
            secondary_input_df["zscore"]=stats.zscore(secondary_input_df[col])

        outlier_percentage = round(len(input_df[input_df["zscore"].abs() >= thres]) / len(input_df) * 100, 1)

        if outlier_percentage >= 10:
            print_and_log(f'[ OUTLIER ]: {col}: {outlier_percentage}. DELETING COLUMN!!!', 'YELLOW')
            input_df[col] = 1
            if len(secondary_input_df) > 0:
                secondary_input_df[col] = 1
        elif outlier_percentage > 0:
            print_and_log(f'[ OUTLIER ]: {col}: {outlier_percentage}. Converting outliers to missing.', '')
            input_df[col] = np.where(input_df["zscore"].abs() >= thres, np.nan, input_df[col])
            if len(secondary_input_df) > 0:
                secondary_input_df[col] = np.where(secondary_input_df["zscore"].abs() >= 3, np.nan, secondary_input_df[col])
        else:
            pass

        del input_df["zscore"]
        if len(secondary_input_df) > 0:
            del secondary_input_df["zscore"]
    return input_df, secondary_input_df
