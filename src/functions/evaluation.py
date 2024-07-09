from __future__ import print_function

import logging
import os
import shutil
import sys
from os import listdir, remove
from os.path import join, isfile

import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

import definitions
from docxtpl import DocxTemplate
from mailmerge import MailMerge
from optbinning.scorecard import plot_cap
from sklearn import metrics
import scikitplot as skplot

from src import print_and_log

use_mlflow = False
try:
    import mlflow

    use_mlflow = True
    mlflow.set_tracking_uri(definitions.mlflow_tracking_uri)
except:
    pass


def log_graph_to_mlflow(run_id, graph, mlflow_path="graphs"):
    """
    Log a graph artifact to MLflow for the specified run.

    Parameters:
        run_id: str, ID of the MLflow run

        graph: str, path to the graph artifact

        mlflow_path: str, path within MLflow to log the artifact (default is "graphs")

    Returns:
        None
    """
    mlflow.start_run(run_id=run_id)
    mlflow.log_artifact(graph, mlflow_path)
    mlflow.end_run()


def save_graph(graph, session_id_folder, tpl, run_id, doc_file, load_png_from_train=False, train_session_to_eval=None):
    """
        Save a graph to a document file after logging it to MLflow.
            doc_file: str, path to the document file
            
            load_png_from_train: str path to a png file from a TRAIN session
            
            train_session_to_eval: used with combination with load_png_from_train, it's a string that shows the dir to the train session
                                   
        Returns:
            None
        """
    # Insert graphs
    if load_png_from_train:
        context = {}
        old_im = graph

        # Example could be: ./sessions/TRAIN_bg_stage2_2024-07-02 10:03:12.127382_no_tag/auc_graph_xgb.png
        source_file = "./sessions/" + train_session_to_eval + '/' + graph

        # New filename
        new_graph_name = "cost_graph.png"

        # Example format: '/home/mandalorian/Projects/jizzmo/sessions/EVAL_bg_stage2_2024-07-02 11:47:37.433418_no_tag/cost_graph.png'
        destination_file = os.path.join(session_id_folder, new_graph_name)
        # Try to copy the png to EVAL session and handle exceptions
        try:
            shutil.copy(source_file, destination_file)
            print(f"File copied and renamed successfully to {destination_file}")
        except FileNotFoundError as e:
            print(f"Error: Source file not found - {e}")
        except PermissionError as e:
            print(f"Error: Permission denied - {e}")
        except Exception as e:  # Catch-all for other errors
            print(f"An error occurred: {e}")

        new_im = destination_file
        # Log img to mlflow
        log_graph_to_mlflow(run_id,  new_im)

        # Replace old graph with new graph in the document
        tpl.replace_pic(old_im, new_im)
        tpl.render(context)
        tpl.save(doc_file)
        plt.clf()

        # Remove the graph file
        # if os.path.isfile(session_id_folder + '/' + graph + '.png'):  # Commented out for debug
        #     remove(session_id_folder + '/' + graph + '.png')
        print(f"[ EVAL ] Graph {graph} ready")
    else:
        # Load graphs from EVAL session
        context = {}
        old_im = graph
        new_im = session_id_folder + '/' + graph + '.png'

        # Log img to mlflow
        log_graph_to_mlflow(run_id, session_id_folder + '/' + graph + '.png')

        # Replace old graph with new graph in the document
        tpl.replace_pic(old_im, new_im)
        tpl.render(context)
        tpl.save(doc_file)
        plt.clf()

        # Remove the graph file
        # if os.path.isfile(session_id_folder + '/' + graph + '.png'):  # Commented out for debug
        #     remove(session_id_folder + '/' + graph + '.png')
        print(f"[ EVAL ] Graph {graph} ready")

def merge_word(project_name,
               input_data_folder_name,
               input_data_project_folder,
               session_to_eval,
               session_folder_name,
               session_id_folder,
               criterion_column,
               observation_date_column,
               columns_to_exclude,
               periods_to_exclude,
               t1df_period,
               t2df_period,
               t3df_period,
               model_arg,
               missing_treatment,
               params):
    """
    Merge the evaluation results into a Word document.

    Steps:
        1. Set up the MLflow experiment and retrieve the parent and child run IDs.
        2. Load the input data and the data from the training session.

        3. Concatenate the training, test, and time-based datasets into a single dataset.

        4. Load the models, correlation features, and missing values data.

        5. Load the evaluation template document.

        6. Determine the chosen model and whether the problem is multiclass.

        7. Fill in the fields in the evaluation template document.

        8. Write the output document and create a DocxTemplate object.

        9. Generate and save the graphs, inserting them into the document.

        10. Save the final document.

    Parameters:
        project_name (str): The name of the project.

        input_data_folder_name (str): The name of the input data folder.

        input_data_project_folder (str): The name of the input data project folder.

        session_to_eval (str): The name of the session to evaluate.

        session_folder_name (str): The name of the session folder.

        session_id_folder (str): The name of the session ID folder.

        criterion_column (str): The name of the criterion column.

        observation_date_column (str): The name of the observation date column.

        columns_to_exclude (list): A list of columns to exclude.

        periods_to_exclude (list): A list of periods to exclude.

        t1df_period (str): The period for the t1df dataset.

        t2df_period (str): The period for the t2df dataset.

        t3df_period (str): The period for the t3df dataset.

        model_arg (str): The model argument.

        missing_treatment (str): The missing data treatment.

        params (dict): A dictionary of parameters.


    Returns:
        None
    """
    experiment = mlflow.set_experiment(definitions.mlflow_prefix + "_" + project_name)
    parent_run_id = ""
    run_id = ""

    # Find parent run ID
    for result in mlflow.search_runs(experiment.experiment_id).iterrows():
        # @debug
        print("@--------DEBUG_ML_FLOW_RUN_NAME--------@")
        print(f"Session to eval: {session_to_eval}")
        print(f"Run Name: {result[1]['tags.mlflow.runName']}")

        if result[1]['tags.mlflow.runName'] == session_to_eval:
            parent_run_id = result[1]['run_id']
            break
    if parent_run_id == "": # Handle case where parent run is not found
        print_and_log(f"ERROR: No parent run for {session_to_eval} found, aborting", "RED")
        sys.exit()

    child_runs = mlflow.search_runs(experiment.experiment_id,
                                    filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' and tags.mlflow.runName = '{model_arg}'")
    if len(child_runs) > 0:
        run_id = child_runs.run_id.values[0]
    else:
        print_and_log(f"ERROR: Could not find child runs for {model_arg} in session {session_to_eval}", "RED")
        sys.exit()

    # Load train session data
    onlyfiles = [f for f in listdir(input_data_folder_name + input_data_project_folder + '/') if
                 isfile(join(input_data_folder_name + input_data_project_folder + '/', f))]
    if len(onlyfiles) == 0:
        print_and_log('ERROR: No files in input folder. Aborting the program.', "RED")
        sys.exit()

    if 'dict' in onlyfiles[0]:
        dict_file = onlyfiles[0]
        input_file = onlyfiles[1]
    else:
        input_file = onlyfiles[0]

    _, _, extension = str(input_file).partition('.')
    # if 'csv' not in extension:
    #
    #    print_and_log('ERROR: input data not a csv file.', "RED")
    #    sys.exit()

    try:
        df_total = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file)
    except:
        df_total = pd.read_parquet(input_data_folder_name + input_data_project_folder + '/' + input_file)

    """if len(df_total.columns.to_list()) == 1:
        df_total = pd.read_csv(input_data_folder_name + input_data_project_folder + '/' + input_file, sep=';')
        if len(df_total.columns.to_list()) == 1:
            print_and_log('ERROR: input data separator not any of the following ,;', "RED")
            sys.exit()"""

    # Data loading -------------------------------------------------------------------------------------------
    # Load train session data
    df_train_X = pd.read_feather(session_folder_name + session_to_eval + '/df_x_train.feather')
    df_test_X = pd.read_feather(session_folder_name + session_to_eval + '/df_x_test.feather')
    df_t1df = pd.read_feather(session_folder_name + session_to_eval + '/df_t1df.feather')
    df_t2df = pd.read_feather(session_folder_name + session_to_eval + '/df_t2df.feather')
    df_t3df = pd.read_feather(session_folder_name + session_to_eval + '/df_t3df.feather')

    # Combine all datasets for the total scope
    df_total_scope = df_train_X.append(df_test_X, ignore_index=True)
    df_total_scope = df_total_scope.append(df_t1df, ignore_index=True)
    df_total_scope = df_total_scope.append(df_t2df, ignore_index=True)
    df_total_scope = df_total_scope.append(df_t3df, ignore_index=True)
    print("[ EVAL ] Data loaded")

    # Create a combined train and test dataset, adding a 'Dataset' label
    keys = ['Train', 'Test']
    dfs = [df_train_X, df_test_X]
    df_train_test = pd.concat([df.assign(Dataset=key) for key, df in zip(keys, dfs)])

    # Load evaluation-related data
    models = pd.read_csv(session_folder_name + session_to_eval + '/models.csv')
    corr_feat = pd.read_csv(session_folder_name + session_to_eval + '/' + model_arg + '/correl_features.csv').set_index('Unnamed: 0')
    corr_feat_raw = pd.read_csv(session_folder_name + session_to_eval + '/' + model_arg + '/correl_raw_features.csv').set_index('Unnamed: 0')
    features = corr_feat.columns

    missing_table = pd.read_csv('./output_data/' + input_data_project_folder + '/missing_values.csv')
    if model_arg == 'lr':
        lr_table = pd.read_csv(session_folder_name + session_to_eval + '/' + model_arg + '/lr_table.csv')

    # Load evaluation template doc
    template = "./template.docx"
    document = MailMerge(template)

    # Choose a model
    if model_arg == 'dt':
        chosen_model = 'Decision Tree Classifier'
    elif model_arg == 'xgb':
        chosen_model = 'XGBoost Classifier'
    elif model_arg == 'rf':
        chosen_model = 'Random forest Classifier'
    elif model_arg == 'lr':
        chosen_model = 'Logistic regression'

    # Add multiclass flag used mainly in plotting below
    is_multiclass = False

    # Check if we have multiclass classification
    if df_total[criterion_column].nunique() > 2:
        is_multiclass = True

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
        statistical_tool=str('Python, Sklearn (Random Forest, Decision tree), Xgboost, Logistic regression'),
        nb_of_bands=0 if is_multiclass else str(df_train_X[model_arg + '_y_pred_prob'].nunique()),
        list_of_variables=str(np.unique(corr_feat.index.values)),
        list_of_raw_variables=str(np.unique(corr_feat_raw.index.values)),
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

    print("[ EVAL ] Starting graphs")

    """
        - Graphs 1-3 and 1.1-3.1: These plot the evolution of the criterion rate (average or sum/count)
        over time for both binary and multiclass cases, using different datasets (df_total and df_total_scope).
        
        - Graph 4 & 5: Creates heatmaps to visualize correlations between features, using seaborn's heatmap function.
        
        - Graph 5.1: Shows correlations between features and the criterion variable with bar plots.
        
        - Graph 6.0: Generates a table for LR model to show the coefficients and error terms of features,
         and optionally describes these features with a dictionary.
         
        - Graph 6, 6.1, 7, 8, 8.1, 8.2: Generates bar plots showing the evolution of the criterion rate,
         number of cases, and share of bands over time.
         
        - Graph 9, 9.1, 9.2, 9.3, 9.4: Generates graphs to visualize the model's performance,
         such as ROC curves and CAP curves.
         
        - Graph 10, 10.1, 10.2, 10.3: Plot distributions and statistics related to deciles of predicted probabilities.
        
        - Graph 11, 11.1: Displays the evolution of the criterion rate by deciles,
         optionally showing a secondary criterion rate if available.
         
        - Graph 12: If not in multiclass mode, this generates a table of predictor and the share of the predicted class.
        
        - Graph 13: Displays missing values or a logo if no missing values are present.
    
    """
    # Graph X  ---------------------------------------------------------------------------------------------------
    # graph = 'graphX'
    # try:
    #     for file in os.listdir("./sessions/" + session_to_eval):
    #
    #         # Plot cost_graph
    #         if file == "cost_graph_" + model_arg + ".png":
    #             cost_graph_name = "cost_graph_" + model_arg + ".png"
    #             save_graph(cost_graph_name,
    #                        session_id_folder,
    #                        tpl,
    #                        run_id,
    #                        DEST_FILE,
    #                        load_png_from_train=True,
    #                        train_session_to_eval=session_to_eval)
    #
    #         # Plot error_graph
    #         elif file == "error_graph_" + model_arg + ".png":
    #             cost_graph_name = "error_graph_" + model_arg + ".png"
    #             save_graph(cost_graph_name,
    #                        session_id_folder,
    #                        tpl,
    #                        run_id,
    #                        DEST_FILE,
    #                        load_png_from_train=True,
    #                        train_session_to_eval=session_to_eval)
    #
    # except Exception as e:
    #     print(f"[ GRAPH ERROR ] Could not load model auc and error graphs: {e}")
    #     pass
    # Graph 0 ---------------------------------------------------------------------------------------------------
    graph = 'graph0'
    df_to_export = models[models['Method'] == model_arg].drop_duplicates()
    dfi.export(df_to_export,
               session_id_folder + '/' + graph + '.png',
               max_rows=None,
               max_cols=None,
               table_conversion="matplotlib")

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 1 ---------------------------------------------------------------------------------------------------
    # Precompute statistics for Graphs 1-3
    agg_data = pd.DataFrame()
    if is_multiclass:
        agg_data = df_total[[criterion_column, observation_date_column]].groupby(
            [observation_date_column, criterion_column]).agg(counts=(criterion_column, 'count'), ).reset_index()
        totals = agg_data.groupby([observation_date_column])['counts'].sum().reset_index().rename(
            columns={"counts": "total"})
        agg_data = agg_data.merge(totals, on=observation_date_column, how='left')
        agg_data['percentage'] = agg_data['counts'] / agg_data['total']

        # TODO: Does it make sense in Multiclass?
    plot = []
    graph = "graph1"
    if not is_multiclass:
        plot = df_total[[criterion_column, observation_date_column]].groupby(observation_date_column).mean().plot(
            kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10), title=graph)
        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)
    else:
        import matplotlib.ticker as mticker
        plot = agg_data.set_index(([observation_date_column, criterion_column])).percentage.unstack().plot(
            kind='bar', figsize=(15, 10), ylabel='Average Criterion Rate', title=graph)

        # plot.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 2 ---------------------------------------------------------------------------------------------------
    # TODO: Does it make sense in Multiclass?
    graph = "graph2"
    if not is_multiclass:

        plot = df_total[[criterion_column, observation_date_column]].groupby(observation_date_column).sum().plot(
            kind='bar', ylabel='NB Criterion cases', figsize=(15, 10), title=graph)
    else:
        plot = agg_data.set_index(([observation_date_column, criterion_column])).counts.unstack().plot(
            kind='bar', figsize=(15, 10), ylabel='NB Criterion cases', title=graph)

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)
    # Graph 3 ---------------------------------------------------------------------------------------------------
    # TODO: Does it make sense in Multiclass?
    graph = 'graph3'
    if not is_multiclass:
        plot = df_total[[criterion_column, observation_date_column]].groupby(observation_date_column).count().plot(
            kind='bar', ylabel='NB Total cases', figsize=(15, 10), title=graph)
    else:
        plot = agg_data[[observation_date_column, 'total']].plot(kind='bar', x=observation_date_column, y='total',
                                                                 figsize=(25, 10), ylabel='NB Total cases', title=graph)

    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)
    del agg_data

    # Graph 1.1 ---------------------------------------------------------------------------------------------------
    # TODO: Does it make sense in Multiclass?
    graph = 'graph1.1'
    plot = df_total_scope[[criterion_column, observation_date_column]].groupby(observation_date_column).mean().plot(
        kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10), title=graph)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 2.1 ---------------------------------------------------------------------------------------------------
    # TODO: Does it make sense in Multiclass?
    graph = 'graph2.1'
    plot = df_total_scope[[criterion_column, observation_date_column]].groupby(observation_date_column).sum().plot(
        kind='bar', ylabel='NB Criterion cases', figsize=(15, 10), title=graph)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 3.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph3.1'
    plot = df_total_scope[[criterion_column, observation_date_column]].groupby(observation_date_column).count().plot(
        kind='bar', ylabel='NB Total cases', figsize=(15, 10), title=graph)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 4 ---------------------------------------------------------------------------------------------------
    graph = 'graph4'

    # corr_feat = corr_feat.set_index('Unnamed: 0')

    plot = sns.heatmap(corr_feat, annot=True,
                xticklabels=corr_feat.columns,
                yticklabels=corr_feat.columns,
                linewidths=.1)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 5 ---------------------------------------------------------------------------------------------------
    graph = 'graph5'

    # corr_feat_raw = corr_feat_raw.set_index('Unnamed: 0')

    plot = sns.heatmap(corr_feat_raw, annot=True,
                xticklabels=corr_feat_raw.columns,
                yticklabels=corr_feat_raw.columns,
                linewidths=.1)
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 5.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph5.1'
    # plot = correl_feat_y[['Unnamed: 0', '0']].groupby('Unnamed: 0').sum().plot(kind='bar', ylabel='Correlation of the features vs Criterion Rate', figsize=(15, 10))
    # features = corr_feat.columns

    plot = df_train_X[features].corrwith(df_train_X[criterion_column], method='pearson').plot(kind='bar',
                                                                                              ylabel='Correlation of the features vs Criterion Rate',
                                                                                              figsize=(15, 10))
    fig = plot.get_figure()
    fig.savefig(session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 6.0 ---------------------------------------------------------------------------------------------------
    graph = 'graph6.0'
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    grid = pd.DataFrame()

    if model_arg == 'lr':
        features = list(lr_table['index'].values)

    print(f"Size of dataset: {df_train_X.shape}")   # debug

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
            prefix = el
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
         'criterion_rate', 'coef', 'error']]    # 10 features

    try:
        dictionary = pd.read_csv(input_data_folder_name + input_data_project_folder + '/dict.csv', sep=';')
        print('Dictionary file found!')

        def describe(row):
            mouchard_col = row['mouchard_name']

            # If 'const', no description will be added
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

        # debug time
        from timeit import default_timer as timer
        time_start = timer()

        # Add column 'Description' to grid and grid_raw. This is also observed to take some time
        grid['Description'] = grid.apply(describe, axis=1)

        time_end = timer()                                                                              # debug
        print(f"Time needed to apply: {time_end - time_start:.2f} for dataset of size {grid.shape}")    # debug

        grid_raw['Description'] = grid.apply(describe, axis=1)

    except Exception as e:
        print(e)
        logging.info('No dictionary found. Columns wont be described in the grid!')

    print(grid)
    try:    # Saves graph
        dfi.export(grid, session_id_folder + '/' + graph + '.png', table_conversion="matplotlib", max_rows=-1)
        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    except Exception as e:
        print(e)
        print_and_log("WARNING: Some of the predictors are RAW and with too many categories. Exclude them!", "YELLOW")

        pass

    bands_column = model_arg + '_bands_predict_proba'
    # Graph 6 ---------------------------------------------------------------------------------------------------
    graph = 'graph6'
    if not is_multiclass:
        if params["secondary_criterion_columns"]:

            secondary_col1 = params["secondary_criterion_columns"][0]
            secondary_col2 = params["secondary_criterion_columns"][1]

            print('Columns for secondary response rate:', secondary_col1, secondary_col2)

            ax_left = plt.subplot(1, 2, 1)
            ax_right = plt.subplot(1, 2, 2)

            plt.subplot(1, 2, 1)

            df_train_X[[criterion_column, bands_column]].groupby(bands_column).mean().plot(kind='bar', figsize=(15, 5),
                                                                                           linewidth=0.1, stacked=True,
                                                                                           ax=ax_left, title=graph)
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
                                                                                                ax=ax_right, title=graph)
            plt.xlabel('Bands')
            plt.legend(shadow=True)
            plt.ylabel('Secondary Criterion rate')


        else:
            plot = df_train_X[[criterion_column, bands_column]].groupby(bands_column).mean().plot(
                kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10), title=graph)

        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 6.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph6.1'
    if not is_multiclass:
        plot = df_train_X[[criterion_column, bands_column]].groupby(bands_column).count().plot(
            kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10), title=graph)
        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 7 ---------------------------------------------------------------------------------------------------
    graph = 'graph7'
    if not is_multiclass:
        plot = pd.crosstab([df_train_test['Dataset'], df_train_test[criterion_column]], df_train_test[bands_column],
                           margins=True).style.background_gradient()
        dfi.export(plot, session_id_folder + '/' + graph + '.png', max_rows=-1, max_cols=-1,
                   table_conversion="matplotlib")

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 8 ---------------------------------------------------------------------------------------------------
    # TODO cache cross-tabs by using multiple aggfuncs
    graph = 'graph8'
    if not is_multiclass:
        plot = pd.crosstab(df_total_scope[observation_date_column], df_total_scope[bands_column],
                           values=df_total_scope[criterion_column],
                           aggfunc='mean').plot(
            kind='bar', ylabel='Average Criterion Rate', figsize=(15, 10), edgecolor='white',
            linewidth=0.2)
        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 8.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph8.1'
    if not is_multiclass:
        plot = pd.crosstab(df_total_scope[observation_date_column], df_total_scope[bands_column],
                           values=df_total_scope[criterion_column],
                           aggfunc='count').plot(
            kind='bar', ylabel='Nb of cases per bands per observation period', figsize=(15, 10), edgecolor='white',
            linewidth=0.2)
        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 8.2 ---------------------------------------------------------------------------------------------------
    graph = 'graph8.2'
    if not is_multiclass:
        plot = pd.crosstab(df_total_scope[observation_date_column], df_total_scope[bands_column],
                           normalize='index').plot(
            kind='bar', ylabel='Share of bands per observation period', figsize=(15, 10), edgecolor='white',
            linewidth=0.2, stacked=True)
        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    if is_multiclass:
        encodings = pd.read_csv("output_data/" + project_name + '/' + 'encoded_labels.csv').to_dict()

    model_arg_y_pred_prob = model_arg + '_y_pred_prob'

    # Graph 9 ---------------------------------------------------------------------------------------------------
    graph = 'graph9'
    if not is_multiclass:
        df = df_total_scope[
            [criterion_column, model_arg_y_pred_prob]]  # .sort_values(by=[bands_column], ascending=True)

        fig = sns.displot(df, x=model_arg_y_pred_prob, hue=criterion_column, kind="ecdf")
        # fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 9.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph9.1'
    if not is_multiclass:
        df = df_train_X[[criterion_column, model_arg_y_pred_prob]]  # .sort_values(by=[bands_column], ascending=True)

        fpr, tpr, threshold = metrics.roc_curve(df[criterion_column], df[model_arg_y_pred_prob])
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

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)
    else:
        graph = 'graph9.1'
        y_proba = df_train_X[df_train_X.columns[df_train_X.columns.str.contains('proba')]]
        y_train = df_train_X[criterion_column].apply(lambda x: encodings['class_label'][x])
        skplot.metrics.plot_roc(y_train, y_proba, figsize=(10, 6))

        plt.savefig(session_id_folder + "/" + graph + '.png')
        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 9.2 ---------------------------------------------------------------------------------------------------
    graph = 'graph9.2'
    if not is_multiclass:
        df = df_test_X[[criterion_column, model_arg_y_pred_prob]]  # .sort_values(by=[bands_column], ascending=True)

        fpr, tpr, threshold = metrics.roc_curve(df[criterion_column], df[model_arg_y_pred_prob])
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

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)
    else:
        y_proba = df_test_X[df_test_X.columns[df_test_X.columns.str.contains('proba')]]
        y_test = df_test_X[criterion_column].apply(lambda x: encodings['class_label'][x])
        skplot.metrics.plot_roc(y_test, y_proba, figsize=(10, 6))

        plt.savefig(session_id_folder + "/" + graph + '.png')
        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 9.3 ---------------------------------------------------------------------------------------------------  
    graph = 'graph9.3'
    if not is_multiclass:
        df = df_train_X[[criterion_column, model_arg_y_pred_prob]]  # .sort_values(by=[bands_column], ascending=True)
        plot_cap(df[criterion_column], df[model_arg_y_pred_prob])

        plt.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 9.4 ---------------------------------------------------------------------------------------------------
    graph = 'graph9.4'
    if not is_multiclass:
        df = df_test_X[[criterion_column, model_arg_y_pred_prob]]  # .sort_values(by=[bands_column], ascending=True)
        plot_cap(df[criterion_column], df[model_arg_y_pred_prob])

        plt.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 10 ---------------------------------------------------------------------------------------------------
    graph = 'graph10'
    if not is_multiclass:
        plot = df_train_X.copy()
        plot['deciles'] = pd.qcut(plot[model_arg_y_pred_prob], 10, duplicates='drop')
        plot = plot[[criterion_column, 'deciles']].groupby('deciles').count().plot(kind='bar',
                                                                                   ylabel='Nb of cases in each Proba',
                                                                                   figsize=(15, 10), edgecolor='white',
                                                                                   linewidth=0.2, title=graph)

        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 10.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph10.1'
    if not is_multiclass:
        plot = df_test_X.copy()
        plot['deciles'] = pd.qcut(plot[model_arg_y_pred_prob], 10, duplicates='drop')
        plot = plot[[criterion_column, 'deciles']].groupby('deciles').count().plot(kind='bar',
                                                                                   ylabel='Nb of cases in each Proba',
                                                                                   figsize=(15, 10), edgecolor='white',
                                                                                   linewidth=0.2, title=graph)

        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 10.2 ---------------------------------------------------------------------------------------------------
    graph = 'graph10.2'
    if not is_multiclass:
        accumulation_points = (
                df_train_X[model_arg_y_pred_prob].round(3).value_counts() / df_train_X[
            model_arg_y_pred_prob].count()).astype(
            float).round(2)
        accumulation_points.round(2).head(5).plot(kind='bar', figsize=(15, 5), linewidth=0.1, stacked=True,
                                                  title='TOP5 Accumulation points by probability')

        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 10.3 ---------------------------------------------------------------------------------------------------
    graph = 'graph10.3'
    if not is_multiclass:
        accumulation_points = (
                df_test_X[model_arg_y_pred_prob].round(3).value_counts() / df_test_X[
            model_arg_y_pred_prob].count()).astype(
            float).round(2)
        accumulation_points.round(2).head(5).plot(kind='bar', figsize=(15, 5), linewidth=0.1, stacked=True,
                                                  title='TOP5 Accumulation points by probability')

        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 11 ---------------------------------------------------------------------------------------------------
    graph = 'graph11'
    if not is_multiclass:
        temp_df = df_train_X.copy()
        temp_df['deciles'] = pd.qcut(temp_df[model_arg_y_pred_prob], 10, duplicates='drop')

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
                                                                                  linewidth=0.2, ax=ax_left, title=graph)
            plt.title(
                'Criterion Rate evolution - primary as described in the document and secondary - {} \ {}'.format(
                    secondary_col1, secondary_col2))
            plt.legend(shadow=True)
            plt.ylabel('Primary Criterion rate')

            plt.subplot(1, 2, 2)
            temp_df["secondary_criterion"] = temp_df[secondary_col1] / temp_df[secondary_col2]
            temp_df[["secondary_criterion", 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                                       ylabel='Nb of cases in each Proba',
                                                                                       figsize=(15, 10),
                                                                                       edgecolor='white',
                                                                                       linewidth=0.2, ax=ax_right, title=graph)
            plt.xlabel('Bands')
            plt.legend(shadow=True)
            plt.ylabel('Secondary Criterion rate')


        else:
            plot[[criterion_column, 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                               ylabel='Nb of cases in each Proba',
                                                                               figsize=(15, 10), edgecolor='white',
                                                                               linewidth=0.2, title=graph)

        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 11.1 ---------------------------------------------------------------------------------------------------
    graph = 'graph11.1'
    if not is_multiclass:
        temp_df = df_test_X.copy()
        temp_df['deciles'] = pd.qcut(temp_df[model_arg_y_pred_prob], 10, duplicates='drop')

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
                                                                                  linewidth=0.2, ax=ax_left, title=graph)
            plt.title(
                'Criterion Rate evolution - primary as described in the document and secondary - {} \ {}'.format(
                    secondary_col1, secondary_col2))
            plt.legend(shadow=True)
            plt.ylabel('Primary Criterion rate')

            plt.subplot(1, 2, 2)
            temp_df["secondary_criterion"] = temp_df[secondary_col1] / temp_df[secondary_col2]
            temp_df[["secondary_criterion", 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                                       ylabel='Nb of cases in each Proba',
                                                                                       figsize=(15, 10),
                                                                                       edgecolor='white',
                                                                                       linewidth=0.2, ax=ax_right, title=graph)
            plt.xlabel('Bands')
            plt.legend(shadow=True)
            plt.ylabel('Secondary Criterion rate')


        else:
            plot[[criterion_column, 'deciles']].groupby('deciles').mean().plot(kind='bar',
                                                                               ylabel='Nb of cases in each Proba',
                                                                               figsize=(15, 10), edgecolor='white',
                                                                               linewidth=0.2, title=graph)

        fig = plot.get_figure()
        fig.savefig(session_id_folder + '/' + graph + '.png')

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 12 ---------------------------------------------------------------------------------------------------
    graph = 'graph12'
    if not is_multiclass:
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

        save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph 13 ---------------------------------------------------------------------------------------------------
    graph = 'graph13'

    if len(missing_table[missing_table['percent_missing'] > 0]) > 0:
        dfi.export(missing_table[missing_table['percent_missing'] > 0].head(100),
                   session_id_folder + '/' + graph + '.png', max_rows=-1, max_cols=-1, table_conversion="matplotlib")

    else:
        shutil.copy('gizmo_logo.png', session_id_folder + '/' + graph + '.png')

    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)

    # Graph appendix loop  ---------------------------------------------------------------------------------------------------
    # todo: fix. Const something
    i = 1
    # TODO: We have graphs untill 10_a
    for el in features:
        try:
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
                plt.title(f"{graph}")
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
                    temp_df[['deciles_ratio', col_y2]].groupby('deciles_ratio').mean().plot(kind='bar',
                                                                                            figsize=(15, 20),
                                                                                            linewidth=0.1,
                                                                                            ax=ax_left_5)
                    plt.xlabel(f'Graph 9 - Mean Criterion rate Deciles based on the ratio')

                    plt.ylabel('Criterion rate')

                    plt.subplot(5, 2, 10)
                    temp_df[['deciles_ratio', col_y2]].groupby('deciles_ratio').count().plot(kind='bar',
                                                                                             figsize=(15, 20),
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
                try:
                    # fig = plot.get_figure()
                    fig = plt.gcf()
                    fig.savefig(session_id_folder + '/' + graph + '.png')
                    save_graph(graph, session_id_folder, tpl, run_id, DEST_FILE)
                    plt.close(fig)
                except Exception as e:
                    print(e)


                # # Insert graphs
                #
                # context = {}
                # old_im = graph
                # new_im = session_id_folder + '/' + graph + '.png'
                #
                # tpl.replace_pic(old_im, new_im)
                # tpl.render(context)
                # tpl.save(DEST_FILE)
                # plt.clf()
                # print(f"[ EVAL ] Graph {graph} ready")
        except Exception as e:
            # TODO graph_a_31 exception prints constantly
            print(f"[ Graphs error ] {e}")
            pass

    for file in os.listdir(session_id_folder):
        if file.endswith('.png'):
            os.remove(session_id_folder + '/' + file)
