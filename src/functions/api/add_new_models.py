import json
import os
import subprocess
from datetime import datetime

from flask import jsonify
import definitions


def add_new_models(new_models_winners):
    """
    Add new models based on the provided winners.

    Args:
        new_models_winners (dict): A dictionary containing project names as keys and model combinations as values.

    Returns:
        flask.Response: Response object indicating the success or failure of adding new models.
    """
    try:
        grid_bash_modification(new_models_winners)
        predict_bash_modification(new_models_winners)
        update_param_files(new_models_winners)

        subprocess.call([f"/bin/bash", f"{definitions.ROOT_DIR}/run_gridsearch_winners.sh"])

        response = jsonify('New models added successfully!')
        response.status_code = 200
        return response
    except Exception as e:
        print(e)
        response = jsonify(f'ERROR: {e}')
        return response
    finally:
        pass


def update_param_files(new_models_winners):
    """
    Update parameter files with new model versions based on the provided winners.

    Args:
        new_models_winners (dict): A dictionary containing project names as keys and model combinations as values.
    """
    params_path = f"{definitions.EXTERNAL_DIR}/params/"
    file_list = os.listdir(params_path)
    today = datetime.today().strftime("%Y%m%d")

    for proj in new_models_winners:
        for file in file_list:
            if proj in file:
                with open(os.path.join(params_path + file), 'r', encoding='utf-8') as param_file:
                    json_object = json.load(param_file)
                    json_object['model_version'] = f"{proj}_v{today}"
                with open(os.path.join(params_path + file), 'w', encoding='utf-8') as output_param_file:
                    json.dump(json_object, output_param_file)


def grid_bash_modification(new_models_winners):
    """
   Modify the grid search bash script based on the new model winners.

   Args:
       new_models_winners (dict): A dictionary containing project names as keys and model combinations as values.
   """
    with open("run_gridsearch_winners.sh", "r") as simu_bash_file:
        header_lines = [next(simu_bash_file) for _ in range(7)]

        for proj in new_models_winners:
            combination = new_models_winners[proj]
            tp, sl, period, _, training_rows, nb_features = combination.split("_")
            header_lines.append(
                f"nice -n 16 python3 -W ignore $DIR/grid_search.py --project {proj} --winner yes "
                f"--tp {tp} --sl {sl} --training_rows {training_rows} --period {period} "
                f"--nb_features {nb_features}\n")
    with open("run_gridsearch_winners.sh", "w") as simu_bash_file:
        for line in header_lines:
            simu_bash_file.write(line)


def predict_bash_modification(new_models_winners):
    """
   Modify the prediction bash script based on the new model winners.

   Args:
       new_models_winners (dict): A dictionary containing project names as keys and model combinations as values.
   """
    with open("run_predict.sh", "r") as simu_bash_file:
        header_lines = [next(simu_bash_file) for _ in range(7)]

        for proj in new_models_winners:
            combination = new_models_winners[proj]
            tp, sl, period, _, training_rows, nb_features = combination.split("_")
            header_lines.append(f"nice -n 10 python3 -W ignore $DIR/main.py --project {proj} --predict_module standard "
                                f"--main_model xgb --pred_data_prep ardi  --tp {tp} --sl {sl} --period {period} "
                                f"--nb_tree_features {nb_features}&\n")
    with open("run_predict.sh", "w") as simu_bash_file:
        for line in header_lines:
            simu_bash_file.write(line)
