import importlib
import json
import pickle
import sys

import pandas as pd

import definitions
from src.classes.DfAggregator import DfAggregator
from src.classes.SessionManager import SessionManager
from src.functions import api_communication
from src.functions.predict import calculate_predictors
from src.functions.printing_and_logging import print_end, print_and_log


class ModuleClass(SessionManager):
    def __init__(self, args):
        SessionManager.__init__(self, args)
        self.main_model = args.main_model.lower()
        self.models_list = ['xgb', 'rf', 'dt']
        self.args = args
        self.defs = json.load(open(f'implemented_models/{self.project_name}/definitions_predict.json'))

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        # sys.stdout = open(os.devnull, 'w')  # forbid print
        # Prepare the input data
        self.loader.data_load_prep(in_data_folder=self.input_data_folder_name,
                                   in_data_proj_folder=self.input_data_project_folder)

        """
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
        print(self.loader.in_df.shape)"""
        self.merging_additional_files_procedure()

        self.loader.in_df = calculate_predictors.calculate_predictors(self.loader.in_df, self.defs)

        # Load models
        for model in self.models_list:
            model_path = f"./implemented_models/{self.project_name}/{model}/model_train.pkl"
            model_pkl = pickle.load(open(model_path, 'rb'))
            self.loader.final_features = model_pkl.final_features

            # Predictions
            self.loader.in_df[f"predict_{model}"] = model_pkl.model.predict_proba(
                self.loader.in_df[self.loader.final_features])[:, 1]
            self.loader.in_df[f"predict_{model}"] = self.loader.in_df[f"predict_{model}"].round(5)

            # todo: add precision score calculations for the last 2800 (dynamic) rows as validation that the model is still performing

            predict_last_row = \
                model_pkl.model.predict_proba(self.loader.in_df[self.loader.final_features].iloc[[-1]])[0][1]
            if predict_last_row < 0.01:
                predict_last_row = 0.01
            predict_last_row = round(float(predict_last_row), 5)
            self.loader.in_df[f"predict_last_{model}"] = predict_last_row
            self.loader.in_df["model"] = self.project_name

        predict_columns = ['time', "criterion_buy", "criterion_sell", "open", "high", "low", "close"]
        for col in self.loader.in_df.columns.tolist():
            if 'predict' in col:
                predict_columns.append(col)
        self.loader.in_df[predict_columns].to_csv(f"./implemented_models/{self.project_name}/predictions.csv",
                                                  index=False)

        if definitions.api_url_post_results_predict:
            api_communication.api_post(definitions.api_url_post_results_predict, self.loader.in_df[predict_columns])

        print_end()
        sys.stdout = sys.__stdout__  # enable print

    def merging_additional_files_procedure(self):
        count = 0

        if self.params["additional_tables"]:
            for file in self.params["additional_tables"]:
                print_and_log(f"[ ADDITIONAL SOURCES ] Merging {file}", "GREEN")

                merge_group_cols = self.params["additional_tables"][file]
                merge_group_cols_input_df = merge_group_cols.copy()
                merge_group_cols_input_df.append(self.params["criterion_column"])
                aggregator = DfAggregator(params=self.params)
                merged_temp = aggregator.aggregation_procedure(
                    df_to_aggregate=self.loader.additional_files_df_dict[count],
                    columns_to_group=merge_group_cols)

                suffix = "_" + file.split('.')[0]
                for el in merged_temp.columns:
                    if el not in merge_group_cols_input_df:
                        merged_temp[el + suffix] = merged_temp[el].copy()
                        del merged_temp[el]

                if len(merged_temp) > 1:
                    periods = ['M', 'T']

                    # if 1!=1:
                    if self.observation_date_column in merge_group_cols:
                        for period in periods:
                            merged_temp[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(merged_temp[self.observation_date_column]).dt.to_period(period)
                            self.loader.in_df[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(self.loader.in_df[self.observation_date_column]).dt.to_period(period)
                            if self.under_sampling: self.loader.in_df_f[self.observation_date_column + '_temp'] = \
                                pd.to_datetime(self.loader.in_df_f[self.observation_date_column]).dt.to_period(period)

                            merge_group_cols_periods = merge_group_cols.copy()
                            merge_group_cols_periods.remove(self.observation_date_column)
                            merge_group_cols_periods.append(self.observation_date_column + '_temp')

                            self.loader.in_df = self.loader.in_df.merge(merged_temp, how='left',
                                                                        on=merge_group_cols,
                                                                        suffixes=("", f"_{period}{suffix}"))
                            missing_cols = self.loader.in_df[
                                self.loader.in_df.columns[self.loader.in_df.isnull().mean() > 0.80]].columns.to_list()
                            self.loader.in_df = self.loader.in_df.drop(columns=missing_cols)

                            if self.under_sampling:
                                self.loader.in_df_f = self.loader.in_df_f.merge(merged_temp,
                                                                                how='left',
                                                                                on=merge_group_cols,
                                                                                suffixes=("",
                                                                                          f"_{period}{suffix}"))
                    else:
                        self.loader.in_df = self.loader.in_df.merge(merged_temp, how='left', on=merge_group_cols,
                                                                    suffixes=("", f"{suffix}"))
                        if self.under_sampling: self.loader.in_df_f = self.loader.in_df_f.merge(merged_temp, how='left',
                                                                                                on=merge_group_cols,
                                                                                                suffixes=(
                                                                                                    "", f"{suffix}"))
                count += 1
