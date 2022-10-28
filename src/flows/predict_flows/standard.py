import importlib
import pickle
import sys

import pandas as pd

import definitions
from src.classes.SessionManager import SessionManager
from src.functions import api_communication
from src.functions.printing_and_logging import print_end, print_and_log


class ModuleClass(SessionManager):
    def __init__(self, args):
        SessionManager.__init__(self, args)
        self.main_model = args.main_model.lower()
        self.models_list = ['xgb', 'rf', 'dt']
        # self.models_list = ['xgb']
        self.args = args
        self.output_df = None

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        # sys.stdout = open(os.devnull, 'w')  # forbid print
        # Prepare the input data
        self.loader.data_load_prep(in_data_folder=self.input_data_folder_name,
                                   in_data_proj_folder=self.input_data_project_folder)

        module_prep = self.args.pred_data_prep.lower()
        module_prep_lib = importlib.import_module(f'src.flows.data_prep_flows.{module_prep}')

        prep = module_prep_lib.ModuleClass(args=self.args)
        prep.loader.in_df = self.loader.in_df.copy()
        prep.loader.additional_files_df_dict = self.loader.additional_files_df_dict
        # prep.predict_session_flag = 1
        prep.run()

        del prep
        del self.loader.additional_files_df_dict
        # Load output data to be ready for prediction
        self.output_df = pd.read_parquet(
            self.output_data_folder_name + self.input_data_project_folder + "/output_data_file.parquet")

        # Load models
        for model in self.models_list:
            model_path = f"./implemented_models/{self.project_name}/{model}/model_train.pkl"
            model_pkl = pickle.load(open(model_path, 'rb'))

            # Predictions
            self.output_df[f"predict_{model}"] = model_pkl.model.predict_proba(
                self.output_df[model_pkl.final_features])[:, 1]
            self.output_df[f"predict_{model}"] = self.output_df[f"predict_{model}"].round(5)

        self.output_df["symbol"] = self.params["symbol"]
        self.output_df["direction"] = self.params["direction"]
        self.output_df["version"] = self.params["model_version"]
        self.output_df["tp"] = self.args.tp
        self.output_df["sl"] = self.args.sl
        self.output_df["period"] = self.args.period
        self.output_df["nb_features"] = self.args.nb_tree_features
        self.output_df["time_stamp"] = str(self.start_time.strftime("%Y-%m-%d %H:%M:%S"))

        predict_columns = ['time', "criterion_buy", "criterion_sell", "open", "high", "low", "close", "symbol",
                           "direction", "version", "tp", "sl", "period", "nb_features", "time_stamp"]
        for col in self.output_df.columns.tolist():
            if 'predict' in col:
                predict_columns.append(col)

        # store results
        self.output_df[predict_columns].to_csv(f"./implemented_models/{self.project_name}/predictions.csv", index=False)
        if definitions.api_url_post_results_predict:
            try:
                api_communication.api_post(definitions.api_url_post_results_predict, self.output_df[predict_columns])
            except Exception as e:
                print(f"ERROR API: {e}")
                pass

        print_end()
        print_and_log(f"{self.output_df.time.max(), self.output_df.time.dtype}", "")
        sys.stdout = sys.__stdout__  # enable print
