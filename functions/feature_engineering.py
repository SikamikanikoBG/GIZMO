import pandas as pd
import logging
import functions
from datetime import datetime
import numpy as np


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
    logging.info('Feat eng: Ratios created')
    return df


def create_tree_feats(df, columns, criterion):
    for col in columns:
        print(f"Trying tree feature for {col}")
        df[col + '_tree'], _ = functions.cut_into_bands(X=df[[col]], y=df[criterion], depth=1)

        if df[col + '_tree'].nunique() == 1:
            del df[col + '_tree']
        else:
            class0_val = str(round(df[df[col + '_tree'] == 0][col].max(), 4))
            df = df.rename(columns={col + "_tree": col + "_tree_" + class0_val})

            print('Feature engineering: New feature added with Decision Tree {}'.format(col + "_tree_" + class0_val))
            logging.info(
                'Feature engineering: New feature added with Decision Tree {}'.format(col + "_tree_" + class0_val))
    logging.info('Feat eng: trees created')
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
        logging.info('Feat eng: keep only columns with correlation > 0.05: %s', corr_cols)
    else:
        a = X.corr()
        time = datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
        if flag_raw == 'yes':
            a.to_csv(session_id_folder + '/' + model_corr + '/correl_raw_features.csv')
        else:
            a.to_csv(session_id_folder + '/' + model_corr + '/correl_features.csv')

    return corr_cols
