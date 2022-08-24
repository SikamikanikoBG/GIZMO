import pandas as pd
import numpy as np
import os
import subprocess
import itertools

import definitions

grid_param_init = {"tp": [0.0025, 0.0040, 0.0060],
              "sl": [0.0020, 0.0040, 0.0060, 0.0080, 0.0100],
              "period": [120, 240, 480, 720]
              }

grid_param = {"tp": [0.0025],
              "sl": [0.0040, 0.0060, 0.0080],
              "period": [240, 480],
              "t_val_size_per_period" : [2800],
              "training_rows": [1400, 2800, 5600, 7000, 10000, 15000, 25000, 35000]
              }

results_df = pd.DataFrame()

def before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows):
    t1 = "2023-08-01 00:00:00"
    t2 = "2023-08-06 00:00:00"
    t3 = "2023-08-12 00:00:00"

    df = pd.read_parquet('./output_data/ardi/output_data_file.parquet')

    if training_rows != False:
        df = df.tail(3*t_val_size_per_period + training_rows).copy()

    df.loc[df.tail(t_val_size_per_period*3).index, 'time'] = np.datetime64(t1)
    df.loc[df.tail(t_val_size_per_period*2).index, 'time'] = np.datetime64(t2)
    df.loc[df.tail(t_val_size_per_period).index, 'time'] = np.datetime64(t3)

    df.to_parquet('./output_data/ardi/output_data_file.parquet')

def load_train(tp, sl, period, t_val_size_per_period, training_rows):
    subprocess.call(["python", "main.py", "--project", "ardi",
                    "--data_prep_module", "ardi", "--tp", str(tp), "--sl", str(sl), "--period", str(period)],
                         stdout=open(f"{definitions.ROOT_DIR}/logs/grid_results.txt", "a"))

    before_train_dedicate_temp_validation_periods(t_val_size_per_period, training_rows)

    tag = f"{str(tp)}_{str(sl)}_{str(period)}"
    subprocess.call(["python", "main.py", "--project", "ardi",
                     "--train_module", "standard", "--tag", tag],
                    stdout=open(f"{definitions.ROOT_DIR}/logs/grid_results.txt", "a"))

    all_subdirs = [f"{definitions.ROOT_DIR}/sessions/{d}" for d in os.listdir(definitions.ROOT_DIR + '/sessions') if os.path.isdir(definitions.ROOT_DIR + '/sessions/' + d)]
    latest_train_session_dir = max(all_subdirs, key=os.path.getmtime)
    models_loop = pd.read_csv(f"{latest_train_session_dir}/models.csv")
    models_loop['combination'] = tag
    return models_loop


a = 0
b = 0
for combination in itertools.product(grid_param["tp"], grid_param["sl"], grid_param["period"],grid_param["t_val_size_per_period"],grid_param["training_rows"]):
    a+=1

for combination in itertools.product(grid_param["tp"], grid_param["sl"], grid_param["period"], grid_param["t_val_size_per_period"], grid_param["training_rows"]):
    b+=1
    print(f"Loop ready {round(b/a,2)*100}%: {b} from total {a} combinations. Combinations: {combination}")
    models_loop_df = load_train(combination[0], combination[1], combination[2], combination[3], combination[4])
    results_df = results_df.append(models_loop_df)

    results_df.to_csv("./sessions/grid_search_results.csv", index=False)


