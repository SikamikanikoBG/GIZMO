import os
import pickle
import sys
from importlib import import_module

from src.classes.SessionManager import SessionManager
from src.functions.printing_and_logging import print_end


class ModuleClass(SessionManager):
    def __init__(self, args):
        SessionManager.__init__(self, args)
        self.main_model = args.main_model.lower()
        self.models_list = ['xgb', 'rf', 'dt']
        self.args = args

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        sys.stdout = open(os.devnull, 'w') # forbid print
        # Prepare the input data
        self.loader.data_load_prep(in_data_folder=self.input_data_folder_name,
                                   in_data_proj_folder=self.input_data_project_folder)

        module_prep = self.args.pred_data_prep.lower()
        module_prep_lib = import_module(f'src.flows.data_prep_flows.{module_prep}')
        prep = module_prep_lib.ModuleClass(args=self.args)
        prep.loader.in_df = self.loader.in_df.copy()
        prep.loader.additional_files_df_dict = self.loader.additional_files_df_dict
        prep.predict_session_flag = 1
        prep.run()

        # Load output data to be ready for prediction
        self.loader.data_load_train(output_data_folder_name=self.output_data_folder_name,
                                    input_data_project_folder=self.input_data_project_folder)

        # Load models
        for model in self.models_list:
            model_path = f"./implemented_models/{self.project_name}/{model}/model_train.pkl"
            model_pkl = pickle.load(open(model_path, 'rb'))
            self.loader.final_features = model_pkl.final_features

            # Predictions
            self.loader.in_df[f"predict_{model}"] = model_pkl.model.predict_proba(self.loader.in_df[self.loader.final_features])[:, 1]
            self.loader.in_df[f"predict_{model}"] = self.loader.in_df[f"predict_{model}"].round(5)
            predict_last_row = model_pkl.model.predict_proba(self.loader.in_df[self.loader.final_features].iloc[[-1]])[0][1]
            if predict_last_row < 0.01:
                predict_last_row = 0.01
            predict_last_row = round(float(predict_last_row), 5)
            self.loader.in_df[f"predict_last_{model}"] = predict_last_row

        predict_columns = ['time', "criterion_buy", "criterion_sell", "open", "high", "low", "close"]
        for col in self.loader.in_df.columns.tolist():
            if 'predict' in col:
                predict_columns.append(col)
        self.loader.in_df[predict_columns].to_csv(f"./implemented_models/{self.project_name}/predictions.csv", index=False)

        print_end()
        sys.stdout = sys.__stdout__  # enable print