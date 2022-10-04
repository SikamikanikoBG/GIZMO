import os
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import definitions


def before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows, project):
    t1 = "2023-08-01 00:00:00"
    t2 = "2023-08-06 00:00:00"
    t3 = "2023-08-12 00:00:00"

    df = pd.read_parquet(f'./output_data/{project}/output_data_file.parquet')

    if training_rows:
        df = df.tail(3 * t_val_size_per_period + training_rows).copy()

    df.loc[df.tail(t_val_size_per_period * 3).index, 'time'] = np.datetime64(t1)
    df.loc[df.tail(t_val_size_per_period * 2).index, 'time'] = np.datetime64(t2)
    df.loc[df.tail(t_val_size_per_period).index, 'time'] = np.datetime64(t3)

    df.to_parquet(f'./output_data/{project}/output_data_file.parquet')


def load_train(tp, sl, period, t_val_size_per_period, training_rows, nb_tree_features, project):
    subprocess.call(["python", "main.py", "--project", f"{project}",
                     "--data_prep_module", "ardi", "--tp", str(tp), "--sl", str(sl), "--period", str(period)],
                    stdout=open(f"{definitions.ROOT_DIR}/logs/grid_results_{project}.txt", "a"))

    before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows, project)

    tag = f"{str(tp)}_{str(sl)}_{str(period)}_{str(t_val_size_per_period)}_{str(training_rows)}_{str(nb_tree_features)}"
    subprocess.call(["python", "main.py", "--project", f"{project}",
                     "--train_module", "standard", "--tag", tag, "--nb_tree_features", str(nb_tree_features)],
                    stdout=open(f"{definitions.ROOT_DIR}/logs/grid_results_{project}.txt", "a"))

    all_subdirs = [f"{definitions.ROOT_DIR}/sessions/{d}" for d in os.listdir(definitions.ROOT_DIR + '/sessions') if
                   os.path.isdir(definitions.ROOT_DIR + '/sessions/' + d)]
    latest_train_session_dir = max(all_subdirs, key=os.path.getmtime)

    try:
        models_loop = pd.read_csv(f"{latest_train_session_dir}/models.csv")
        models_loop['combination'] = tag
        models_loop['time'] = datetime.now().strftime('%Y%m%d')
        shutil.rmtree(latest_train_session_dir)
    except Exception as e:
        models_loop = pd.DataFrame()
        models_loop['combination'] = f"UNSUCCESSFUL: {tag}_{e}"

    return models_loop
