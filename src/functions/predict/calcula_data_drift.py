import numpy as np
import pandas as pd


def calculate_data_drift(selected_proj):
    # load data
    path = f"./implemented_models/{selected_proj}"
    df_train = pd.read_feather(f"{path}/df_x_train.feather")
    df_feature_importance_xgb = pd.read_csv(f"{path}/xgb/feat_importance.csv")
    # df_feature_importance_rf = pd.read_csv(f"{path}/rf/feat_importance.csv")
    # df_feature_importance_dt = pd.read_csv(f"{path}/dt/feat_importance.csv")
    df_feature_importance = df_feature_importance_xgb.copy()
    df_predict = pd.read_csv(f"{path}/predictions.csv")

    features_list = df_feature_importance["columns"].unique().tolist()

    # calculate means of the features
    df_feature_importance["mean_train"] = 0
    df_feature_importance["mean_predict"] = 0
    df_feature_importance["data_drift"] = 0

    for feat in features_list:
        mean_train = df_train[feat].mean()
        mean_predict = df_predict[feat].mean()
        data_drift = abs((mean_predict / mean_train) - 1)
        df_feature_importance["mean_train"] = np.where(df_feature_importance["columns"] == feat, mean_train,
                                                       df_feature_importance["mean_train"])
        df_feature_importance["mean_predict"] = np.where(df_feature_importance["columns"] == feat, mean_predict,
                                                         df_feature_importance["mean_predict"])
        df_feature_importance["data_drift"] = np.where(df_feature_importance["columns"] == feat, data_drift,
                                                       df_feature_importance["data_drift"])

    # Metrics
    df_feature_importance["data_drift_mult_importances"] = df_feature_importance["data_drift"] * df_feature_importance[
        "importances"]
    mean_drift = round(df_feature_importance["data_drift"].mean(), 2)
    mean_drift_weighted = round(
        df_feature_importance["data_drift_mult_importances"].sum() / df_feature_importance["importances"].sum(), 2)

    # top 5 metrics
    df_feature_importance_top5 = df_feature_importance.head().copy()
    df_feature_importance_top5["data_drift_mult_importances"] = df_feature_importance_top5["data_drift"] * \
                                                                df_feature_importance_top5[
                                                                    "importances"]
    mean_drift_top5 = round(df_feature_importance_top5["data_drift"].mean(), 2)
    mean_drift_weighted_top5 = round(
        df_feature_importance["data_drift_mult_importances"].sum() / df_feature_importance["importances"].sum(), 2)

    return mean_drift, mean_drift_weighted, mean_drift_top5, mean_drift_weighted_top5
