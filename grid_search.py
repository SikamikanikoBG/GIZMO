import argparse
from  datetime import datetime
import itertools
import os
import subprocess

import numpy as np
import pandas as pd

import definitions

grid_param_init = {"tp": [0.0025, 0.0040, 0.0060],
                   "sl": [0.0020, 0.0040, 0.0060, 0.0080, 0.0100],
                   "period": [120, 240, 480, 720],
                   "t_val_size_per_period": [],
                   "training_rows": []
                   }

grid_param = {"tp": [0.0025, 0.0040],
              "sl": [0.0040, 0.0080],
              "period": [720],
              "t_val_size_per_period": [2800],
              "training_rows": [7000, 15000],
              "nb_features": [10, 30, 50, 70, 100]
              }

results_df = pd.DataFrame()

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('--project', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
args = parser.parse_args()
project = args.project.lower()


def before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows):
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


def load_train(tp, sl, period, t_val_size_per_period, training_rows, nb_tree_features):
    subprocess.call(["python", "main.py", "--project", f"{project}",
                     "--data_prep_module", "ardi", "--tp", str(tp), "--sl", str(sl), "--period", str(period)],
                    stdout=open(f"{definitions.ROOT_DIR}/logs/grid_results_{project}.txt", "a"))

    before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows)

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
    except Exception as e:
        models_loop = pd.DataFrame()
        models_loop['combination'] = f"UNSUCCESSFUL: {tag}_{e}"
    return models_loop


a = 0
b = 0
for combination in itertools.product(grid_param["tp"], grid_param["sl"], grid_param["period"],
                                     grid_param["t_val_size_per_period"], grid_param["training_rows"], grid_param["nb_features"]):
    a += 1

for combination in itertools.product(grid_param["tp"], grid_param["sl"], grid_param["period"],
                                     grid_param["t_val_size_per_period"], grid_param["training_rows"], grid_param["nb_features"]):
    b += 1
    time_start = datetime.now()

    models_loop_df = load_train(combination[0], combination[1], combination[2], combination[3], combination[4], combination[5])
    results_df = results_df.append(models_loop_df)

    time_end = datetime.now()
    time = time_end - time_start
    time_remaining = (a-b)*time
    print(f"Loop ready {round(b / a, 2) * 100}%: {b} from total {a} combinations. Combinations: {combination}. "
          f"Elapsed time: {time} minutes. Time remaining: {time_remaining}")

    results_df.to_csv(f"./sessions/grid_search_results_{project}.csv", index=False)
