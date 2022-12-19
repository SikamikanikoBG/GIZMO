import argparse
import subprocess
from datetime import datetime
import itertools
import pandas as pd
import definitions
from notify_run import Notify
notify = Notify()

from src.functions import api_communication
from src.functions.grid_search.grid_search_functions import load_train

results_df = pd.DataFrame()

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--project', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
parser.add_argument('--winner', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
parser.add_argument('--tp', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
parser.add_argument('--sl', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
parser.add_argument('--training_rows', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
parser.add_argument('--nb_features', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
parser.add_argument('--period', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
args = parser.parse_args()
project = args.project.lower()
winner = args.winner
training_rows = int(args.training_rows)
nb_features = int(args.nb_features)
tp = float(args.tp)
sl = float(args.sl)
period = int(args.period)

if winner:
    grid_param = {"tp": [tp],
                  "sl": [sl],
                  "period": [period],
                  "t_val_size_per_period": [1300],
                  "training_rows": [training_rows],
                  "nb_features": [nb_features]
                  }
else:
    grid_param = {"tp": [0.0025],
                  "sl": [0.0150],
                  "period": [240, 480],
                  "t_val_size_per_period": [1300],
                  "training_rows": [4000, 7000],
                  "nb_features": [15, 30, 50]
                  }

a, b = 0, 0
for combination in itertools.product(grid_param["tp"], grid_param["sl"], grid_param["period"],
                                     grid_param["t_val_size_per_period"], grid_param["training_rows"],
                                     grid_param["nb_features"]):
    a += 1

for combination in itertools.product(grid_param["tp"], grid_param["sl"], grid_param["period"],
                                     grid_param["t_val_size_per_period"], grid_param["training_rows"],
                                     grid_param["nb_features"]):
    b += 1
    time_start = datetime.now()

    if combination[1] > combination[0]:
        models_loop_df, times_list = load_train(combination[0], combination[1], combination[2], combination[3], combination[4],
                                    combination[5], project, winner)

        models_loop_df['currency'] = project

        # todo: develop in sql db and remove the comments
        models_loop_df['progress'] = round(b / a, 2) * 100
        results_df = results_df.append(models_loop_df)

        results_df.to_csv(f"{definitions.EXTERNAL_DIR}/sessions/grid_search_results_{project}.csv", index=False)

        if definitions.api_url_post_models_simulations and not winner:
            try:
                models_loop_df = models_loop_df[~models_loop_df["DataSet"].isin(['test_X', 'train_X'])].copy()
                api_communication.api_post(definitions.api_url_post_models_simulations, models_loop_df)
            except:
                pass
    else:
        times_list = []
        pass

    time_end = datetime.now()
    time = time_end - time_start
    time_remaining = (a - b) * time
    msg = f"[ {project} ] Loop ready {round(b / a, 2) * 100}%: {b} from total {a} combinations. " \
          f"Combinations: {combination}. Elapsed time: {time} minutes. Time remaining: {time_remaining}. Subtimes: {times_list}"
    print(msg)
if winner:
    msg = f"[ {project} ] Winner ready and saved in implementation folder."
    subprocess.call(["curl", definitions.notifications_url_grid, "-d", f'"{msg}"'])
    #notify.send(message=f"[ {project} ] Winner ready and saved in implementation folder.")
else:
    msg = f"[ {project} ] Simulation ready. Go and check it on ArDi Report! :)"
    subprocess.call(["curl", definitions.notifications_url_grid, "-d", f'"{msg}"'])
    #notify.send(message=f"[ {project} ] Simulation ready. Go and check it on ArDi Report! :)")

