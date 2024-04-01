import os
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import definitions


def before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows, project):
    """
    Update time column in the DataFrame to dedicate temporary validation periods.

    Steps:
    1. Define temporary validation periods t1, t2, and t3.
    2. Read the DataFrame from the output data file.
    3. Trim the DataFrame based on training rows if provided.
    4. Convert the 'time' column to datetime format.
    5. Assign temporary validation periods to the 'time' column based on the tail indices.
    6. Save the updated DataFrame to the output data file.

    Parameters:
    - t_val_size_per_period: int, size of each temporary validation period
    - training_rows: int, number of training rows to keep
    - project: str, project name

    Returns:
    - None
    """
    t1 = "2022-08-01 00:00:00"
    t2 = "2022-08-06 00:00:00"
    t3 = "2022-08-12 00:00:00"

    df = pd.read_parquet(f'{definitions.ROOT_DIR}/output_data/{project}/output_data_file.parquet')

    if training_rows:
        df = df.tail(3 * t_val_size_per_period + training_rows).copy()

    df['time'] = pd.to_datetime(df['time'])
    df.loc[df.tail(t_val_size_per_period * 3).index, 'time'] = np.datetime64(t1)
    df.loc[df.tail(t_val_size_per_period * 2).index, 'time'] = np.datetime64(t2)
    df.loc[df.tail(t_val_size_per_period).index, 'time'] = np.datetime64(t3)

    df.to_parquet(f'{definitions.ROOT_DIR}/output_data/{project}/output_data_file.parquet')


def load_train(tp, sl, period, t_val_size_per_period, training_rows, nb_tree_features, project, winner):
    """
    Load and train a model for the specified project.

    Steps:
    1. Record the start time.
    2. Open a log file for recording results.
    3. Call the main script with project details for data preparation.
    4. Record the preparation time.
    5. Create temporary validation periods using dedicated function.
    6. Generate a tag based on input parameters.
    7. Call the main script for training with project details and tag.
    8. Record the training time.
    9. Obtain the latest training session directory.
    10. Process the results of the session and handle winner scenario.
    11. Calculate preparation, training, and folder processing times.
    12. Return the models loop DataFrame and a list of times.

    Parameters:
    - tp: int, tp parameter
    - sl: int, sl parameter
    - period: int, period parameter
    - t_val_size_per_period: int, size of each temporary validation period
    - training_rows: int, number of training rows to keep
    - nb_tree_features: int, number of tree features
    - project: str, project name
    - winner: bool, flag indicating if the model is a winner

    Returns:
    - models_loop: DataFrame, results of the training session
    - times_list: list, times for preparation, training, and folder processing
    """
    time_start = datetime.now()
    file = open(f"{definitions.EXTERNAL_DIR}/logs/grid_results_{project}.txt", mode="w+")
    subprocess.call(["python", "main.py", "--project", f"{project}",
                     "--data_prep_module", "ardi", "--tp", str(tp), "--sl", str(sl), "--period", str(period)],
                    stdout=open(f"{definitions.EXTERNAL_DIR}/logs/grid_results_{project}.txt", "a"))
    time_prep = datetime.now()

    # Create temporal validation periods
    before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows, project)

    tag = f"{str(tp)}_{str(sl)}_{str(period)}_{str(t_val_size_per_period)}_{str(training_rows)}_{str(nb_tree_features)}"
    subprocess.call(["python", "main.py", "--project", f"{project}",
                     "--train_module", "standard", "--tag", tag, "--nb_tree_features", str(nb_tree_features)],
                    stdout=open(f"{definitions.EXTERNAL_DIR}/logs/grid_results_{project}.txt", "a"))
    time_train = datetime.now()

    all_subdirs = [f"{definitions.EXTERNAL_DIR}/sessions/{d}" for d in os.listdir(definitions.EXTERNAL_DIR + '/sessions') if
                   os.path.isdir(definitions.EXTERNAL_DIR + '/sessions/' + d)]

    # avoid obtaining wrong directory, that belongs to other project. This is caused when there are several grid searches
    # ran in parallel
    for dir in all_subdirs[:]:
        if project not in dir:
            all_subdirs.remove(dir)

    try:
        # Obtain the results of the session
        latest_train_session_dir = max(all_subdirs, key=os.path.getmtime)
        models_loop = pd.read_csv(f"{latest_train_session_dir}/models.csv")
        models_loop['combination'] = tag
        models_loop['time'] = datetime.now().strftime('%Y%m%d')
        if not winner:
            shutil.rmtree(latest_train_session_dir)
        elif winner:
            shutil.move(latest_train_session_dir, f"{definitions.EXTERNAL_DIR}/implemented_models/{project}")
        time_folders = datetime.now()

        prep_time = time_prep - time_start
        training_time = time_train - time_prep
        folders_time = time_folders - time_train

    except Exception as e:
        models_loop = pd.DataFrame()
        models_loop['combination'] = f"UNSUCCESSFUL: {tag}_{e}"
        training_time = 0
        prep_time = 0
        folders_time = 0


    times_list = [prep_time, training_time, folders_time]


    return models_loop, times_list
