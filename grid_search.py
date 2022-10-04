import argparse
from datetime import datetime
import itertools

import pandas as pd

import definitions

from src.functions import api_communication
from src.functions.grid_search.grid_search_functions import load_train

grid_param = {"tp": [0.0025, 0.0040, 0.0060],
              "sl": [0.0040, 0.0060, 0.0080, 0.0100],
              "period": [120, 240, 480],
              "t_val_size_per_period": [1300],
              "training_rows": [4000, 7000, 14000],
              "nb_features": [30, 50]
              }

grid_param_ = {"tp": [0.0025],
              "sl": [0.0080],
              "period": [480],
              "t_val_size_per_period": [1300],
              "training_rows": [4000],
              "nb_features": [30]
              }


results_df = pd.DataFrame()

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--project', type=str,
                    help='name of the project. Should  the same as the input folder and the param file.')
args = parser.parse_args()
project = args.project.lower()

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

    models_loop_df = load_train(combination[0], combination[1], combination[2], combination[3], combination[4],
                                combination[5], project)

    models_loop_df['currency'] = project
    results_df = results_df.append(models_loop_df)

    if definitions.api_url_post_models_simulations:
        try:
            api_communication.api_post(definitions.api_url_post_models_simulations, models_loop_df)
        except:
            pass
    results_df.to_csv(f"./sessions/grid_search_results_{project}.csv", index=False)

    time_end = datetime.now()
    time = time_end - time_start
    time_remaining = (a - b) * time
    print(f"Loop ready {round(b / a, 2) * 100}%: {b} from total {a} combinations. Combinations: {combination}. "
          f"Elapsed time: {time} minutes. Time remaining: {time_remaining}")
