from __future__ import print_function

import logging
import os
import sys
from os import listdir
from os.path import join, isfile

import dataframe_image as dfi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
from docxtpl import DocxTemplate
from mailmerge import MailMerge
from sklearn import metrics
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks
import shutil


def merge_word(input_data_folder_name, input_data_project_folder, session_to_eval, session_folder_name,
               session_id_folder, criterion_column,
               observation_date_column,
               columns_to_exclude,
               periods_to_exclude,
               t1df_period,
               t2df_period,
               t3df_period,
               model_arg,
               missing_treatment, params):
    # Load train session data
    onlyfiles = [f for f in listdir(input_data_folder_name + input_data_project_folder + '/') if
                 isfile(join(input_data_folder_name + input_data_project_folder + '/', f))]
    if len(onlyfiles) == 0:
        logging.error('ERROR: No files in input folder. Aborting the program.')
        sys.exit()

    if 'dict' in onlyfiles[0]:
        dict_file = onlyfiles[0]
        input_file = onlyfiles[1]
    else:
        input_file = onlyfiles[0]

    _, _, extention = str(input_file).partition('.')
    if 'csv' not in extention:
        logging.error('ERROR: input data not a csv file.')
        sys.exit()

    df_total = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file)
    if len(df_total.columns.to_list()) == 1:
        df_total = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file, sep=';')
        if len(df_total.columns.to_list()) == 1:
            logging.error('ERROR: input data separator not any of the following ,;')
            sys.exit()

    df_train_X = pq.read_table(session_folder_name + session_to_eval + '/df_x_train.parquet')
    df_train_X = df_train_X.to_pandas()
    df_test_X = pq.read_table(session_folder_name + session_to_eval + '/df_x_test.parquet')
    df_test_X = df_test_X.to_pandas()
    df_t1df = pq.read_table(session_folder_name + session_to_eval + '/df_t1df.parquet')
    df_t1df = df_t1df.to_pandas()
    df_t2df = pq.read_table(session_folder_name + session_to_eval + '/df_t2df.parquet')
    df_t2df = df_t2df.to_pandas()
    df_t3df = pq.read_table(session_folder_name + session_to_eval + '/df_t3df.parquet')
    df_t3df = df_t3df.to_pandas()

    df_total_scope = df_train_X.append(df_test_X, ignore_index=True)
    df_total_scope = df_total_scope.append(df_t1df, ignore_index=True)
    df_total_scope = df_total_scope.append(df_t2df, ignore_index=True)
    df_total_scope = df_total_scope.append(df_t3df, ignore_index=True)

    keys = ['Train', 'Test']
    dfs = [df_train_X, df_test_X]
    df_train_test = pd.concat([df.assign(Dataset=key) for key, df in zip(keys, dfs)])

    models = pd.read_csv(session_folder_name + session_to_eval + '/models.csv')
    corr_feat = pd.read_csv(session_folder_name + session_to_eval + '/' + model_arg + '/correl_features.csv')
    corr_feat_raw = pd.read_csv(session_folder_name + session_to_eval + '/' + model_arg + '/correl_raw_features.csv')
    correl_feat_y = pd.read_csv('./output_data/' + input_data_project_folder + '/correl.csv')
    correl_feat_y = correl_feat_y[correl_feat_y.iloc[:, 0].isin(corr_feat['Unnamed: 0'].unique().tolist())]
    missing_table = pd.read_csv('./output_data/' + input_data_project_folder + '/missing_values.csv')
    if model_arg == 'lr':
        lr_table = pd.read_csv(session_folder_name + session_to_eval + '/' + model_arg + '/lr_table.csv')

    # Load evaluation template doc
    template = "./template.docx"
    document = MailMerge(template)

    if model_arg == 'dt':
        chosen_model = 'Decision Tree Classifier'
    elif model_arg == 'xgb':
        chosen_model = 'XGBoost Classifier'
    elif model_arg == 'rf':
        chosen_model = 'Random forest Classifier'
    elif model_arg == 'lr':
        chosen_model = 'Logistic regression'

    # Fill in fields
    document.merge(
        df_total_min_date=str(df_total[observation_date_column].min()),
        df_total_max_date=str(df_total[observation_date_column].max()),
        df_train_min_date=str(df_train_X[observation_date_column].min()),
        df_train_max_date=str(df_train_X[observation_date_column].max()),
        df_test_min_date=str(df_test_X[observation_date_column].min()),
        df_test_max_date=str(df_test_X[observation_date_column].max()),
        df_t1df_date=str(df_t1df[observation_date_column].min()),
        df_t2df_date=str(df_t2df[observation_date_column].min()),
        df_t3df_date=str(df_t3df[observation_date_column].min()),
        excluded_periods=str(periods_to_exclude),
        statistical_tool=str('Python, Sklearn(Random Forest, Decision tree), Xgboost, Logistic regression'),
        nb_of_bands=str(df_train_X[model_arg + '_y_pred_prob'].nunique()),
        list_of_variables=str(corr_feat['Unnamed: 0'].unique()),
        list_of_raw_variables=str(corr_feat_raw['Unnamed: 0'].unique()),
        df_train_volume=str(len(df_train_X)),
        df_test_volume=str(len(df_test_X)),
        df_t1df_volume=str(len(df_t1df)),
        df_t2df_volume=str(len(df_t2df)),
        df_t3df_volume=str(len(df_t3df)),
        df_train_criterion=str(round(sum(df_train_X[criterion_column]) / len(df_train_X[criterion_column]), 3)),
        df_test_criterion=str(round(sum(df_test_X[criterion_column]) / len(df_test_X[criterion_column]), 3)),
        df_t1df_criterion=str(round(sum(df_t1df[criterion_column]) / len(df_t1df[criterion_column]), 3)),
        df_t2df_criterion=str(round(sum(df_t2df[criterion_column]) / len(df_t2df[criterion_column]), 3)),
        df_t3df_criterion=str(round(sum(df_t3df[criterion_column]) / len(df_t3df[criterion_column]), 3)),
        missing_treatment=str(missing_treatment),
        chosen_model=chosen_model,
        train_session=session_to_eval

    )

    # Write output
    output_file_name = session_id_folder + '/gizmo_result_' + model_arg + '.docx'
    document.write(output_file_name)
    tpl = DocxTemplate(output_file_name)
    DEST_FILE = output_file_name

    # Graph 0 ---------------------------------------------------------------------------------------------------
    graph0 = 'graph0'

    dfi.export(models[models['Method'] == model_arg], session_id_folder + '/' + graph0 + '.png', max_rows=-1,
               max_cols=-1, table_conversion="matplotlib")

    # Insert graphs

    context = {}
    old_im = graph0
    new_im = session_id_folder + '/' + graph0 + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 1 ---------------------------------------------------------------------------------------------------
    plot = df_total[[criterion_column, observation_date_column]].groupby(observation_date_column).mean().plot(
        kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/graph1.png')

    # Insert graphs

    context = {}
    old_im = "graph1"
    new_im = session_id_folder + '/graph1.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 2 ---------------------------------------------------------------------------------------------------
    plot = df_total[[criterion_column, observation_date_column]].groupby(observation_date_column).sum().plot(
        kind='bar', ylabel='NB Criterion cases', figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/graph2.png')

    # Insert graphs

    context = {}
    old_im = "graph2"
    new_im = session_id_folder + '/graph2.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 3 ---------------------------------------------------------------------------------------------------
    graph = 'graph3'
    plot = df_total[[criterion_column, observation_date_column]].groupby(observation_date_column).count().plot(
        kind='bar', ylabel='NB Total cases', figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 1.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph1.1'
    plot = df_total_scope[[criterion_column, observation_date_column]].groupby(observation_date_column).mean().plot(
        kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 2.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph2.1'
    plot = df_total_scope[[criterion_column, observation_date_column]].groupby(observation_date_column).sum().plot(
        kind='bar', ylabel='NB Criterion cases', figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 3.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph3.1'
    plot = df_total_scope[[criterion_column, observation_date_column]].groupby(observation_date_column).count().plot(
        kind='bar', ylabel='NB Total cases', figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 4 ---------------------------------------------------------------------------------------------------
    graph = 'graph4'

    corr_feat = corr_feat.set_index('Unnamed: 0')

    sns.heatmap(corr_feat, annot=True,
                xticklabels=corr_feat.columns,
                yticklabels=corr_feat.columns,
                linewidths=.1)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 5 ---------------------------------------------------------------------------------------------------
    graph = 'graph5'

    corr_feat_raw = corr_feat_raw.set_index('Unnamed: 0')

    sns.heatmap(corr_feat_raw, annot=True,
                xticklabels=corr_feat_raw.columns,
                yticklabels=corr_feat_raw.columns,
                linewidths=.1)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 5.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph5.1'
    # plot = correl_feat_y[['Unnamed: 0', '0']].groupby('Unnamed: 0').sum().plot(kind='bar', ylabel='Correlation of the features vs Criterion Rate', figsize=(15, 10))
    features = corr_feat.columns

    plot = df_train_X[features].corrwith(df_train_X[criterion_column], method='pearson').plot(kind='bar',
                                                                                              ylabel='Correlation of the features vs Criterion Rate',
                                                                                              figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 6.0 ---------------------------------------------------------------------------------------------------
    graph0 = 'graph6.0'
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    grid = pd.DataFrame()

    if model_arg == 'lr':
        features = list(lr_table['index'].values)

    for el in features:
        temp = pd.crosstab(df_train_X[el], df_train_X[criterion_column], margins=True)
        share = pd.crosstab(df_train_X[el], df_train_X[criterion_column], margins=True, normalize=True)
        share = share[share.columns[2:]]
        share = share.rename(columns={"All": "Share"})
        temp = pd.concat([temp, share], axis=1)
        temp['feature'] = el
        temp['criterion_rate'] = round(temp[1] / (temp[0] + temp[1]), 2)
        temp = temp.sort_values(by=['criterion_rate'])
        temp = temp.rename(columns={1: "Nb_positive_cases", 0: 'Nb_negative_cases', 'All': 'Total cases'})
        temp['values'] = temp.index
        if 'dummie_' in el:
            prefix, match, sufix = str(el).partition('_dummie_')
        elif 'tree' in el:
            prefix, match, sufix = str(el).partition('_tree')
            sufix = '> ' + sufix
        else:
            sufix = ''
            prefix = 'const'
        if '_binned' in prefix:
            prefix = prefix.replace('_binned', '')
        if '_div_ratio_' in prefix:
            prefix = prefix.replace('_div_ratio_', ' / ')
        temp['cut_off_value'] = sufix
        temp['mouchard_name'] = prefix
        temp['cut_off_value'] = temp['cut_off_value'].str.replace('_', ' ')
        grid = grid.append(temp)

    grid = grid[grid['values'] != 'All']
    grid.reset_index(drop=True, inplace=True)
    grid = round(grid, 2)
    if model_arg == 'lr':
        lr_table = lr_table.rename(columns={'index': 'feature'})
        grid = pd.merge(grid, lr_table, on='feature', how='inner')

    grid.loc[grid['values'] == 0, ['cut_off_value', 'coef', 'error']] = "All other values", '0', '0'

    grid_raw = grid.copy()
    grid = grid[
        ['mouchard_name', 'values', 'cut_off_value', 'Nb_negative_cases', 'Nb_positive_cases', 'Total cases', 'Share',
         'criterion_rate', 'coef', 'error']]

    try:

        dictionary = pd.read_csv(input_data_folder_name + input_data_project_folder + '/dict.csv', sep=';')
        print('Dictionary file found!')

        def describe(row):
            mouchard_col = row['mouchard_name']

            if 'const' in mouchard_col:
                val = ''
            elif '/' in mouchard_col:
                prefix, delim, suffix = str(mouchard_col).partition('/')
                try:
                    col1 = dictionary['Description'].loc[dictionary['Variable'] == str(prefix).replace(' ', '')].values[
                        0]
                except:
                    col1 = 'Missing description'
                try:
                    col2 = dictionary['Description'].loc[dictionary['Variable'] == str(suffix).replace(' ', '')].values[
                        0]
                except:
                    col2 = 'Missing description'
                val = col1 + ' / ' + col2

            else:
                try:
                    col1 = dictionary['Description'].loc[dictionary['Variable'] == mouchard_col].values[0]
                except:
                    col1 = 'Missing description'
                val = col1
            return val

        grid['Description'] = grid.apply(describe, axis=1)
        grid_raw['Description'] = grid.apply(describe, axis=1)
    except Exception as e:
        print(e)
        logging.info('No dictionary found. Columns wont be described in the grid!')
    dfi.export(grid, session_id_folder + '/' + graph0 + '.png', table_conversion="matplotlib")

    # Insert graphs

    context = {}
    old_im = graph0
    new_im = session_id_folder + '/' + graph0 + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 6 ---------------------------------------------------------------------------------------------------
    graph = 'graph6'
    bands_column = model_arg + '_bands_predict_proba'
    if params["secondary_criterion_columns"]:

        secondary_col1 = params["secondary_criterion_columns"][0]
        secondary_col2 = params["secondary_criterion_columns"][1]

        print('Columns for secondary response rate:', secondary_col1, secondary_col2)

        ax_left = plt.subplot(1, 2, 1)
        ax_right = plt.subplot(1, 2, 2)

        plt.subplot(1, 2, 1)

        df_train_X[[criterion_column, bands_column]].groupby(bands_column).mean().plot(kind='bar', figsize=(15, 5),
                                                                                       linewidth=0.1, stacked=True,
                                                                                       ax=ax_left)
        plt.title(
            'Average Criterion Rate evolution - primary as described in the document and secondary - {} \ {}'.format(
                secondary_col1, secondary_col2))
        plt.legend(shadow=True)
        plt.ylabel('Primary Criterion rate')

        plt.subplot(1, 2, 2)
        df_train_X["secondary_criterion"] = df_train_X[secondary_col1] / df_train_X[secondary_col2]
        df_train_X[["secondary_criterion", bands_column]].groupby(bands_column).mean().plot(kind='bar',
                                                                                            figsize=(
                                                                                                15, 5),
                                                                                            linewidth=0.1,
                                                                                            ax=ax_right)
        plt.xlabel('Bands')
        plt.legend(shadow=True)
        plt.ylabel('Secondary Criterion rate')


    else:
        plot = df_train_X[[criterion_column, bands_column]].groupby(bands_column).mean().plot(
            kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10))

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 6.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph6.1'
    bands_column = model_arg + '_bands_predict_proba'
    plot = df_train_X[[criterion_column, bands_column]].groupby(bands_column).count().plot(
        kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 7 ---------------------------------------------------------------------------------------------------
    graph = 'graph7'
    plot = pd.crosstab([df_train_test['Dataset'], df_train_test[criterion_column]], df_train_test[bands_column],
                       margins=True).style.background_gradient()
    dfi.export(plot, session_id_folder + '/' + graph + '.png', max_rows=-1, max_cols=-1, table_conversion="matplotlib")

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 8 ---------------------------------------------------------------------------------------------------
    graph = 'graph8'
    bands_column = model_arg + '_bands_predict_proba'

    plot = pd.crosstab(df_total_scope[observation_date_column], df_total_scope[bands_column],
                       values=df_total_scope[criterion_column],
                       aggfunc='mean').plot(
        kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10), edgecolor='white',
        linewidth=0.2)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 8.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph8.1'
    bands_column = model_arg + '_bands_predict_proba'
    plot = pd.crosstab(df_total_scope[observation_date_column], df_total_scope[bands_column],
                       values=df_total_scope[criterion_column],
                       aggfunc='count').plot(
        kind='bar', ylabel='Nb of cases per bands per observation period', figsize=(15, 10), edgecolor='white',
        linewidth=0.2)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 8.2 ---------------------------------------------------------------------------------------------------
    graph = 'graph8.2'
    bands_column = model_arg + '_bands_predict_proba'

    plot = pd.crosstab(df_total_scope[observation_date_column], df_total_scope[bands_column],
                       normalize='index').plot(
        kind='bar', ylabel='Share of bands per observation period', figsize=(15, 10), edgecolor='white',
        linewidth=0.2, stacked=True)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 9 ---------------------------------------------------------------------------------------------------
    graph = 'graph9'
    bands_column = model_arg + '_y_pred_prob'
    df = df_total_scope[[criterion_column, bands_column]]  # .sort_values(by=[bands_column], ascending=True)

    fig = sns.displot(df, x=bands_column, hue=criterion_column, kind="ecdf")
    # fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 9.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph9.1'
    bands_column = model_arg + '_y_pred_prob'
    df = df_train_X[[criterion_column, bands_column]]  # .sort_values(by=[bands_column], ascending=True)

    fpr, tpr, threshold = metrics.roc_curve(df[criterion_column], df[bands_column])
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 9.2 ---------------------------------------------------------------------------------------------------
    graph = 'graph9.2'
    bands_column = model_arg + '_y_pred_prob'
    df = df_test_X[[criterion_column, bands_column]]  # .sort_values(by=[bands_column], ascending=True)

    fpr, tpr, threshold = metrics.roc_curve(df[criterion_column], df[bands_column])
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 9.3 ---------------------------------------------------------------------------------------------------
    graph = 'graph9.3'
    bands_column = model_arg + '_y_pred_prob'
    df = df_train_X[[criterion_column, bands_column]]  # .sort_values(by=[bands_column], ascending=True)

    plot_cap(df[criterion_column], df[bands_column])

    plt.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 9.4 ---------------------------------------------------------------------------------------------------
    graph = 'graph9.4'
    bands_column = model_arg + '_y_pred_prob'
    df = df_test_X[[criterion_column, bands_column]]  # .sort_values(by=[bands_column], ascending=True)

    plot_cap(df[criterion_column], df[bands_column])

    plt.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 10 ---------------------------------------------------------------------------------------------------
    graph = 'graph10'
    bands_column = model_arg + '_y_pred_prob'
    plot = df_train_X.copy()
    plot['deciles'] = pd.qcut(plot[bands_column], 10, duplicates='drop')
    plot = plot[[criterion_column, 'deciles']].groupby('deciles').count().plot(kind='bar',
                                                                               ylabel='Nb of cases in each Proba',
                                                                               figsize=(15, 10), edgecolor='white',
                                                                               linewidth=0.2)

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 10.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph10.1'
    bands_column = model_arg + '_y_pred_prob'
    plot = df_test_X.copy()
    plot['deciles'] = pd.qcut(plot[bands_column], 10, duplicates='drop')
    plot = plot[[criterion_column, 'deciles']].groupby('deciles').count().plot(kind='bar',
                                                                               ylabel='Nb of cases in each Proba',
                                                                               figsize=(15, 10), edgecolor='white',
                                                                               linewidth=0.2)

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 10.2 ---------------------------------------------------------------------------------------------------
    graph = 'graph10.2'
    model_arg_y_pred_prob = model_arg + "_y_pred_prob"
    accumulation_points = (
            df_train_X[model_arg_y_pred_prob].round(3).value_counts() / df_train_X[
        model_arg_y_pred_prob].count()).astype(
        float).round(2)
    accumulation_points.round(2).head(5).plot(kind='bar', figsize=(15, 5), linewidth=0.1, stacked=True,
                                              title='TOP5 Accumulation points by probability')

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 10.3 ---------------------------------------------------------------------------------------------------
    graph = 'graph10.3'
    accumulation_points = (
            df_test_X[model_arg_y_pred_prob].round(3).value_counts() / df_test_X[model_arg_y_pred_prob].count()).astype(
        float).round(2)
    accumulation_points.round(2).head(5).plot(kind='bar', figsize=(15, 5), linewidth=0.1, stacked=True,
                                              title='TOP5 Accumulation points by probability')

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 11 ---------------------------------------------------------------------------------------------------
    graph = 'graph11'
    bands_column = model_arg + '_y_pred_prob'
    temp_df = df_train_X.copy()
    temp_df['deciles'] = pd.qcut(temp_df[bands_column], 10, duplicates='drop')

    if params["secondary_criterion_columns"]:

        secondary_col1 = params["secondary_criterion_columns"][0]
        secondary_col2 = params["secondary_criterion_columns"][1]

        print('Columns for secondary response rate:', secondary_col1, secondary_col2)

        ax_left = plt.subplot(1, 2, 1)
        ax_right = plt.subplot(1, 2, 2)

        plt.subplot(1, 2, 1)

        temp_df[[criterion_column, 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                              ylabel='Nb of cases in each Proba',
                                                                              figsize=(15, 10), edgecolor='white',
                                                                              linewidth=0.2, ax=ax_left)
        plt.title(
            'Criterion Rate evolution - primary as described in the document and secondary - {} \ {}'.format(
                secondary_col1, secondary_col2))
        plt.legend(shadow=True)
        plt.ylabel('Primary Criterion rate')

        plt.subplot(1, 2, 2)
        temp_df["secondary_criterion"] = temp_df[secondary_col1] / temp_df[secondary_col2]
        temp_df[["secondary_criterion", 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                                   ylabel='Nb of cases in each Proba',
                                                                                   figsize=(15, 10), edgecolor='white',
                                                                                   linewidth=0.2, ax=ax_right)
        plt.xlabel('Bands')
        plt.legend(shadow=True)
        plt.ylabel('Secondary Criterion rate')


    else:
        plot[[criterion_column, 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                           ylabel='Nb of cases in each Proba',
                                                                           figsize=(15, 10), edgecolor='white',
                                                                           linewidth=0.2)

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 11 ---------------------------------------------------------------------------------------------------
    graph = 'graph11.1'
    bands_column = model_arg + '_y_pred_prob'
    temp_df = df_test_X.copy()
    temp_df['deciles'] = pd.qcut(temp_df[bands_column], 10, duplicates='drop')

    if params["secondary_criterion_columns"]:

        secondary_col1 = params["secondary_criterion_columns"][0]
        secondary_col2 = params["secondary_criterion_columns"][1]

        print('Columns for secondary response rate:', secondary_col1, secondary_col2)

        ax_left = plt.subplot(1, 2, 1)
        ax_right = plt.subplot(1, 2, 2)

        plt.subplot(1, 2, 1)

        temp_df[[criterion_column, 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                              ylabel='Nb of cases in each Proba',
                                                                              figsize=(15, 10), edgecolor='white',
                                                                              linewidth=0.2, ax=ax_left)
        plt.title(
            'Criterion Rate evolution - primary as described in the document and secondary - {} \ {}'.format(
                secondary_col1, secondary_col2))
        plt.legend(shadow=True)
        plt.ylabel('Primary Criterion rate')

        plt.subplot(1, 2, 2)
        temp_df["secondary_criterion"] = temp_df[secondary_col1] / temp_df[secondary_col2]
        temp_df[["secondary_criterion", 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                                   ylabel='Nb of cases in each Proba',
                                                                                   figsize=(15, 10), edgecolor='white',
                                                                                   linewidth=0.2, ax=ax_right)
        plt.xlabel('Bands')
        plt.legend(shadow=True)
        plt.ylabel('Secondary Criterion rate')


    else:
        plot[[criterion_column, 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                           ylabel='Nb of cases in each Proba',
                                                                           figsize=(15, 10), edgecolor='white',
                                                                           linewidth=0.2)

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 12 ---------------------------------------------------------------------------------------------------
    graph = 'graph12'

    bands_column = model_arg + '_bands_predict_proba'
    grid_describe = pd.DataFrame()

    for el in features:
        temp = pd.crosstab(df_train_X[el], df_train_X[bands_column], normalize='index')
        temp['Predictor'] = el
        temp = temp[temp.index == 1]

        #
        if 'dummie_' in el:
            prefix, match, sufix = str(el).partition('_dummie_')
        elif 'tree' in el:
            prefix, match, sufix = str(el).partition('_tree')
            sufix = '> ' + sufix
        else:
            sufix = ''
            prefix = 'const'
        if '_binned' in prefix:
            prefix = prefix.replace('_binned', '')
        if '_div_ratio_' in prefix:
            prefix = prefix.replace('_div_ratio_', ' / ')
        temp['cut_off_value'] = sufix
        temp['mouchard_name'] = prefix
        temp['cut_off_value'] = temp['cut_off_value'].str.replace('_', ' ')
        #

        grid_describe = grid_describe.append(temp)

    grid_describe.reset_index(drop=True, inplace=True)
    grid_describe.rename_axis("Nb", axis='index', inplace=True)

    cols = list(grid_describe.columns)
    cols = cols[-3:] + cols[:-3]
    grid_describe = grid_describe[cols]
    try:
        grid_describe['Description'] = grid_describe.apply(describe, axis=1)
    except:
        pass

    dfi.export(grid_describe.round(2).style.background_gradient(cmap='RdYlGn', axis=1),
               session_id_folder + '/' + graph + '.png', max_rows=-1, max_cols=-1, table_conversion="matplotlib")

    # Insert graphs

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph 13 ---------------------------------------------------------------------------------------------------
    graph = 'graph13'

    if len(missing_table[missing_table['percent_missing'] > 0]) > 0:
        dfi.export(missing_table[missing_table['percent_missing'] > 0].head(100),
                   session_id_folder + '/' + graph + '.png', max_rows=-1, max_cols=-1, table_conversion="matplotlib")

    else:
        shutil.copy('gizmo_logo.png', session_id_folder + '/' + graph + '.png')

    context = {}
    old_im = graph
    new_im = session_id_folder + '/' + graph + '.png'

    tpl.replace_pic(old_im, new_im)
    tpl.render(context)
    tpl.save(DEST_FILE)
    plt.clf()

    # Graph appendix loop  ---------------------------------------------------------------------------------------------------
    i = 1

    for el in features:
        if 'const' in el:
            pass
        else:
            graph = 'graph_a_' + str(i)
            i += 1

            # ----------------------------------------
            plt.rc('xtick', labelsize=7)
            plt.rc('ytick', labelsize=7)
            plt.xticks(rotation=70)

            ax_left_1 = plt.subplot(5, 2, 1)
            ax_right_1 = plt.subplot(5, 2, 2)
            ax_left_2 = plt.subplot(5, 2, 3)
            ax_right_2 = plt.subplot(5, 2, 4)
            ax_left_3 = plt.subplot(5, 2, 5)
            ax_right_3 = plt.subplot(5, 2, 6)
            ax_left_4 = plt.subplot(5, 2, 7)
            ax_right_4 = plt.subplot(5, 2, 8)
            ax_left_5 = plt.subplot(5, 2, 9)
            ax_right_5 = plt.subplot(5, 2, 10)

            col_x = observation_date_column
            col_y1 = el
            col_y2 = criterion_column

            mouchard_names = grid_raw[grid_raw['feature'] == el]['mouchard_name'].iloc[0]
            try:
                descriptions = grid_raw[grid_raw['feature'] == el]['Description'].iloc[0]
            except:
                pass

            if '/' in mouchard_names:
                raw_col1, _, raw_col2 = str(mouchard_names).partition(' / ')
                try:
                    raw_col1_desc, _, raw_col2_desc = str(descriptions).partition(' / ')
                except:
                    pass
            else:
                raw_col1 = mouchard_names
                try:
                    raw_col1_desc = descriptions
                except:
                    pass
                raw_col2 = ''
                raw_col2_desc = ''

            temp_df = df_total_scope.copy()
            if temp_df[raw_col1].dtype == object:

                pass
            else:
                temp_df['deciles_raw1'] = pd.qcut(temp_df[raw_col1], 10, duplicates='drop')

            # temp_df.set_index(col_x, inplace=True)
            x = temp_df[col_x]
            y1 = temp_df[col_y1]
            y2 = temp_df[col_y2]

            plt.subplot(5, 2, 1)
            pd.crosstab(x, y1, normalize='index').plot(kind='bar', figsize=(5, 10), linewidth=0.1, stacked=True,
                                                       ax=ax_left_1,
                                                       sharey=False)
            plt.legend(shadow=True)
            plt.ylabel('Share of modality')
            try:
                plt.title(
                    f'Feature: {el} \n Raw columns: {raw_col1}, {raw_col2} \n Description: {raw_col1_desc}, {raw_col2_desc}')
            except:
                plt.title(
                    f'Feature: {el} \n Raw columns: {raw_col1}, {raw_col2}')
            plt.xlabel(f'Graph 1 - Evolution of modalities')

            plt.subplot(5, 2, 2)
            (pd.crosstab(x, y1, aggfunc='sum', values=y2) / pd.crosstab(x, y1, aggfunc='count', values=y2)).plot(
                kind='bar',
                ax=ax_right_1)
            plt.xlabel(f'Graph 2 - Evolution of Criterion rate by modalities \n {col_x}')

            plt.legend(shadow=True)
            plt.ylabel('Criterion rate')

            plt.subplot(5, 2, 3)

            if temp_df[raw_col1].dtype == object:

                temp_df[[col_y2, raw_col1]].groupby(raw_col1).mean().plot(kind='bar', figsize=(15, 20),
                                                                          linewidth=0.1,
                                                                          ax=ax_left_2)
            else:
                temp_df[[col_y2, 'deciles_raw1']].groupby('deciles_raw1').mean().plot(kind='bar', figsize=(15, 20),
                                                                                      linewidth=0.1,
                                                                                      ax=ax_left_2)
            plt.legend(shadow=True)
            plt.xlabel(f'Graph 3 - Mean Criterion rate by Deciles \n {raw_col1}')

            plt.ylabel('Criterion rate')

            plt.subplot(5, 2, 4)
            if temp_df[raw_col1].dtype == object:
                #
                temp_df[[col_y2, raw_col1]].groupby(raw_col1).count().plot(kind='bar', figsize=(15, 20),
                                                                           linewidth=0.1,
                                                                           ax=ax_right_2)
            else:
                temp_df[[col_y2, 'deciles_raw1']].groupby('deciles_raw1').count().plot(kind='bar', figsize=(15, 20),
                                                                                       linewidth=0.1,
                                                                                       ax=ax_right_2, legend=None)
            plt.xlabel(f'Graph 4 - Nb of cases by Deciles \n {raw_col1}')

            plt.ylabel('Nb of cases')

            if raw_col2:

                temp_df['deciles_raw2'] = pd.qcut(temp_df[raw_col2], 10, duplicates='drop')
                temp_df['ratio'] = temp_df[raw_col1] / temp_df[raw_col2]

                temp_df['ratio'] = temp_df['ratio'].replace([np.inf, -np.inf], 0)

                temp_df['deciles_ratio'] = pd.qcut(temp_df['ratio'], 20, duplicates='drop')

                plt.subplot(5, 2, 5)
                temp_df[[col_y2, 'deciles_raw2']].groupby('deciles_raw2').mean().plot(kind='bar', figsize=(15, 20),
                                                                                      linewidth=0.1,
                                                                                      ax=ax_left_3)
                plt.xlabel(f'Graph 5 - Mean Criterion rate by Deciles \n {raw_col2}')

                plt.ylabel('Criterion rate')

                plt.subplot(5, 2, 6)
                temp_df[[col_y2, 'deciles_raw2']].groupby('deciles_raw2').count().plot(kind='bar', figsize=(15, 20),
                                                                                       linewidth=0.1,
                                                                                       ax=ax_right_3, legend=None)
                plt.xlabel(f'Graph 6 - Nb of cases by Deciles \n {raw_col2}')

                plt.ylabel('Nb of cases')

                plt.subplot(5, 2, 7)
                temp_df[['ratio', 'deciles_raw1']].groupby('deciles_raw1').mean().plot(kind='bar', figsize=(15, 20),
                                                                                       linewidth=0.1,
                                                                                       ax=ax_left_4)
                plt.xlabel(f'Graph 7 - Mean ratio by Deciles \n {raw_col1}')

                plt.ylabel('Average ratio')

                plt.subplot(5, 2, 8)
                temp_df[['ratio', 'deciles_raw2']].groupby('deciles_raw2').mean().plot(kind='bar', figsize=(15, 20),
                                                                                       linewidth=0.1,
                                                                                       ax=ax_right_4)
                plt.xlabel(f'Graph 8 - Mean ratio by Deciles \n {raw_col2}')

                plt.ylabel('Average ratio')

                plt.subplot(5, 2, 9)
                temp_df[['deciles_ratio', col_y2]].groupby('deciles_ratio').mean().plot(kind='bar', figsize=(15, 20),
                                                                                        linewidth=0.1,
                                                                                        ax=ax_left_5)
                plt.xlabel(f'Graph 9 - Mean Criterion rate Deciles based on the ratio')

                plt.ylabel('Criterion rate')

                plt.subplot(5, 2, 10)
                temp_df[['deciles_ratio', col_y2]].groupby('deciles_ratio').count().plot(kind='bar', figsize=(15, 20),
                                                                                         linewidth=0.1,
                                                                                         ax=ax_right_5)
                plt.xlabel(f'Graph 10 - NB cases of the ratio')

                plt.ylabel('Criterion rate')
            else:
                plt.subplot(5, 2, 5)
                plt.subplot(5, 2, 6)
                plt.subplot(5, 2, 7)
                plt.subplot(5, 2, 8)
                plt.subplot(5, 2, 9)
                plt.subplot(5, 2, 10)

            plt.subplots_adjust(wspace=0.15, hspace=0.7)

            # ----------------------------------------
            fig = plot.get_figure()
            fig.savefig(session_id_folder + '/' + graph + '.png')

            # Insert graphs

            context = {}
            old_im = graph
            new_im = session_id_folder + '/' + graph + '.png'

            tpl.replace_pic(old_im, new_im)
            tpl.render(context)
            tpl.save(DEST_FILE)
            plt.clf()

    for file in os.listdir(session_id_folder):
        if file.endswith('.png'):
            os.remove(session_id_folder + '/' + file)
