import numpy as np


def calculate_predictors(df, final_features_calcs):
    for feature in final_features_calcs:
        if final_features_calcs[feature]["type"] == "ratio":
            df[feature] = df[final_features_calcs[feature]["raw_col_left"]] / df[final_features_calcs[feature]["raw_col_right"]]
        elif final_features_calcs[feature]["type"] == "single":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]] == final_features_calcs[feature]["cutpoint"], 1, 0)
        elif final_features_calcs[feature]["type"] == "less_than":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]] < final_features_calcs[feature]["cutpoint"], 1, 0)
        elif final_features_calcs[feature]["type"] == "greater_than":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]] > final_features_calcs[feature]["cutpoint"], 1, 0)
        elif final_features_calcs[feature]["type"] == "less_eq_to":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]] <= final_features_calcs[feature]["cutpoint"], 1, 0)
        elif final_features_calcs[feature]["type"] == "greater_eq_to":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]] >= final_features_calcs[feature]["cutpoint"], 1, 0)
        elif final_features_calcs[feature]["type"] == "between_left_incl":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]].between(final_features_calcs[feature]["cutpoint"][0], final_features_calcs[feature]["cutpoint"][1], inclusive='left'), 1, 0)
        elif final_features_calcs[feature]["type"] == "between_right_incl":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]].between(final_features_calcs[feature]["cutpoint"][0], final_features_calcs[feature]["cutpoint"][1], inclusive='right'), 1, 0)
        elif final_features_calcs[feature]["type"] == "between_both_incl":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]].between(final_features_calcs[feature]["cutpoint"][0], final_features_calcs[feature]["cutpoint"][1], inclusive='both'), 1, 0)
        elif final_features_calcs[feature]["type"] == "between":
            df[feature] = np.where(df[final_features_calcs[feature]["raw_col"]].between(final_features_calcs[feature]["cutpoint"][0], final_features_calcs[feature]["cutpoint"][1], inclusive='neither'), 1, 0)
        else:
            print(f"ERROR: Unrecognized type for {feature}: {final_features_calcs[feature]['type']}. Correct it and run again the program.")

    return df
