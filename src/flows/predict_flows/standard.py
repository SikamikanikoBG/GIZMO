import importlib
import pickle
import sys

import numpy as np
import pandas as pd

import definitions
from src.classes.SessionManager import SessionManager
from src.functions.predict.calcula_data_drift import calculate_data_drift
from src.functions.printing_and_logging import print_end, print_and_log


class ModuleClass(SessionManager):
    def __init__(self, args):
        """
        Initialize the ModuleClass.

        Args:
            args (object): Arguments object containing necessary parameters.

        Attributes:
            main_model (str): The main model specified in the arguments, converted to lowercase.
            models_list (list): List of models to be used for prediction.
            args (object): Reference to the arguments object.
            output_df (None or DataFrame): DataFrame to store the output data, initialized as None.
        """
        SessionManager.__init__(self, args)
        # Commented out because main_model is not specified
        # self.main_model = args.main_model.lower()
        self.models_list = ['xgb', 'rf', 'dt']
        # self.models_list = ['xgb']
        self.args = args
        self.output_df = None

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.

        This method orchestrates the actions to be performed by the ModuleClass. It includes the following steps:

        1. Load model features by iterating over the models in the models_list and extracting final features from each model.
        2. Prepare input data by setting columns to include based on all final features and loading data using data_load_prep.
        3. Instantiate a data preparation module based on the pred_data_prep argument and run the preparation process.
        4. Load output data for prediction from a parquet file.
        5. Load models from the specified paths and make predictions for each model.
        6. Add model predictions and parameters to the output DataFrame.
        7. Calculate data drift using the calculate_data_drift function.
        8. Save predictions along with data drift indicators as a CSV file.
        9. Optionally, post predictions to an API if the API URL is defined in the definitions.

        Raises:
            Exception: If an error occurs during the execution, it is caught and logged in red color.

        """
        # Load models features
        all_final_features = []
        missing_features_all = []
        for model in self.models_list:
            model_path = f"{self.implemented_folder}{self.project_name}/{model}/model_train.pkl"
            model_pkl = pickle.load(open(model_path, 'rb'))

            for el in model_pkl.final_features:
                if el not in all_final_features:
                    all_final_features.append(el)

        # sys.stdout = open(os.devnull, 'w')  # forbid print
        # Prepare the input data
        self.columns_to_include = all_final_features.copy()
        self.loader.data_load_prep(in_data_folder=self.input_data_folder_name,
                                   in_data_proj_folder=self.input_data_project_folder)

        module_prep = self.args.pred_data_prep.lower()
        module_prep_lib = importlib.import_module(f'src.flows.data_prep_flows.{module_prep}')

        prep = module_prep_lib.ModuleClass(args=self.args)
        prep.columns_to_include = self.columns_to_include
        prep.loader.in_df = self.loader.in_df.copy()
        prep.loader.additional_files_df_dict = self.loader.additional_files_df_dict
        # prep.predict_session_flag = 1
        prep.run()

        del prep
        del self.loader.additional_files_df_dict

        # Load output data to be ready for prediction
        self.output_df = pd.read_parquet(
            self.output_data_folder_name + self.input_data_project_folder + "/output_data_file.parquet")
        print_and_log("[ PREDICT ] Loading output file done", "")
        try:
            # Load models

            for model in self.models_list:
                try:
                    print_and_log(f"[ PREDICT ] Loading model: {model}", "")
                    model_path = f"{self.implemented_folder}{self.project_name}/{model}/model_train.pkl"
                    model_pkl = pickle.load(open(model_path, 'rb'))

                    # Predictions
                    missing_features = []
                    for feat in model_pkl.final_features:
                        if feat not in self.output_df.columns.tolist():
                            missing_features.append(feat)
                            missing_features_all.append(feat)
                    if len(missing_features) > 0:
                        print_and_log(f"[ PREDICT ] Missing features for model {model}: {missing_features}", "RED")
                        self.output_df[f"predict_{model}"] = 0
                    else:
                        if model == 'xgb':
                            self.output_df[f"predict_{model}"] = model_pkl.model.predict_proba(
                            self.output_df[model_pkl.model.get_booster().feature_names])[:, 1]
                        else:
                            self.output_df[f"predict_{model}"] = model_pkl.model.predict_proba(
                                self.output_df[model_pkl.final_features])[:, 1]
                        self.output_df[f"predict_{model}"] = self.output_df[f"predict_{model}"].round(5)
                except Exception as model_error:
                    print_and_log(f"[ PREDICT ] ERROR for {model}: {model_error}", "RED")
                    self.output_df[f"predict_{model}"] = 0

            print_and_log("[ PREDICT ] Model predictions added", "")
            self.output_df["symbol"] = self.params["symbol"]
            self.output_df["direction"] = self.params["direction"]
            self.output_df["version"] = self.params["model_version"]
            self.output_df["flag_trade"] = self.params["flag_trade"]
            self.output_df["tp"] = self.args.tp
            self.output_df["sl"] = self.args.sl
            self.output_df["period"] = self.args.period
            self.output_df["nb_features"] = self.args.nb_tree_features
            self.output_df["time_stamp"] = str(self.start_time.strftime("%Y-%m-%d %H:%M:%S"))

            print_and_log("[ PREDICT ] Model parameters added", "")
            if self.params["signal_trade"]:
                col1_name = self.params["signal_trade"]["col1_name"]
                col2_name = self.params["signal_trade"]["col2_name"]
                col1_value = self.params["signal_trade"]["col1_value"]
                col2_value = self.params["signal_trade"]["col2_value"]
                col2_direction = self.params["signal_trade"]["col2_direction"]
                if col2_name:
                    if "up" in col2_direction:
                        self.output_df["signal_trade"] = np.where((self.output_df[col1_name] >= col1_value)
                                                                  & ((self.output_df[col2_name] * 10000) > np.float(
                            col2_value)), 1, 0)
                    else:
                        self.output_df["signal_trade"] = np.where((self.output_df[col1_name] >= col1_value)
                                                                  & ((self.output_df[col2_name] * 10000) <= np.float(
                            col2_value)), 1, 0)
                else:
                    self.output_df["signal_trade"] = np.where(self.output_df[col1_name] >= col1_value, 1, 0)

            print_and_log("[ PREDICT ] Signal calculated", "")
            predict_columns = ['time', "criterion_buy", "criterion_sell", "open", "high", "low", "close", "symbol",
                               "direction", "version", "tp", "sl", "period", "nb_features", "time_stamp", "flag_trade",
                               "signal_trade", "flag_trend"]
            predict_columns_minimum = predict_columns.copy()
            for col in self.output_df.columns.tolist():
                if 'predict' in col:
                    predict_columns.append(col)

            # columns with features to be used for data drift detection
            predict_columns_data_drift = predict_columns.copy()

            for col in all_final_features:
                if col not in missing_features_all:
                    predict_columns_data_drift.append(col)

            # get data drift
            self.output_df[predict_columns_data_drift].to_csv(
                f"{self.implemented_folder}{self.project_name}/predictions.csv",
                index=False)

            print_and_log("[ PREDICT ] Predictions csv saved", "")

            try:
                mean_data_drift, mean_w_data_drift, mean_data_drift_top5, mean_w_data_drift_top5 = calculate_data_drift(
                    self.project_name)
                self.output_df["mean_data_drift"] = mean_data_drift
                self.output_df["mean_w_data_drift"] = mean_w_data_drift
                self.output_df["mean_data_drift_top5"] = mean_data_drift_top5
                self.output_df["mean_w_data_drift_top5"] = mean_w_data_drift_top5

                predict_columns.append("mean_data_drift")
                predict_columns.append("mean_w_data_drift")
                predict_columns.append("mean_data_drift_top5")
                predict_columns.append("mean_w_data_drift_top5")

                print_and_log("[ PREDICT ] Data drift calculated", "")

                # other indicators
                other_indicators_list = ['open_14_ema', 'rs_14', 'rsi_14', 'wr_14',
                                         'atr_14', 'open_240_ema', 'rs_240', 'rsi_240', 'wr_240', 'atr_240',
                                         'open_480_ema', 'rs_480', 'rsi_480', 'wr_480', 'atr_480', 'close_10_sma',
                                         'close_50_sma', 'dma', 'high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema',
                                         'pdm_14', 'pdi_14', 'pdi', 'mdm', 'mdm_14_ema', 'mdm_14', 'mdi_14', 'mdi', 'dx_14',
                                         'dx', 'adx', 'adxr', 'log-ret', 'macd', 'macds', 'macdh', 'macd_feat',
                                         'macds_feat',
                                         'boll', 'boll_ub', 'boll_lb', 'boll_feat', 'boll_ub_feat', 'boll_lb_feat']
                for ind in other_indicators_list:
                    if ind not in predict_columns_data_drift:
                        predict_columns_data_drift.append(ind)

                # remove last n rows that does not have enough performance period
                rows_to_remove = self.output_df["time"].head(int(self.args.period))
                self.output_df = self.output_df[~self.output_df["time"].isin(rows_to_remove)].copy()


                # store results
                self.output_df[predict_columns_data_drift].to_csv(
                    f"{self.implemented_folder}/{self.project_name}/predictions.csv",
                    index=False)
                print_and_log("[ PREDICT ] Predictions saved as csv, second time", "")

            except:
                # store results
                self.output_df[predict_columns_minimum].to_csv(
                    f"{self.implemented_folder}/{self.project_name}/predictions.csv",
                    index=False)
                print_and_log("[ PREDICT ] Predictions saved as csv, second time", "")

            print_end()
            print_and_log(f"{self.output_df.time.max(), self.output_df.time.dtype}", "")
            sys.stdout = sys.__stdout__  # enable print
        except Exception as e:
            print_and_log(e, "RED")
