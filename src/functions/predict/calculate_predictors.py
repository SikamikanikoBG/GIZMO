import numpy as np


def calculate_predictors(df, final_features_calcs):
    """
    Calculate predictors based on specified calculations for each feature.

    Steps:
    1. Iterate over each feature in the final_features_calcs dictionary.
    2. Based on the type of calculation:
        - For 'ratio': Calculate the ratio of two raw columns and assign the result to the feature.
        - For 'single': Assign 1 if the raw column value matches the cutpoint, else assign 0.
        - For 'less_than': Assign 1 if the raw column value is less than the cutpoint, else assign 0.
        - For 'greater_than': Assign 1 if the raw column value is greater than the cutpoint, else assign 0.
        - For 'less_eq_to': Assign 1 if the raw column value is less than or equal to the cutpoint, else assign 0.
        - For 'greater_eq_to': Assign 1 if the raw column value is greater than or equal to the cutpoint, else assign 0.
        - For 'between_left_incl': Assign 1 if the raw column value is between the cutpoints with left inclusive, else assign 0.
        - For 'between_right_incl': Assign 1 if the raw column value is between the cutpoints with right inclusive, else assign 0.
        - For 'between_both_incl': Assign 1 if the raw column value is between the cutpoints with both inclusive, else assign 0.
        - For 'between': Assign 1 if the raw column value is between the cutpoints with neither inclusive, else assign 0.
        - For unrecognized types: Log an error message.

    Parameters:
    - df: DataFrame, input DataFrame
    - final_features_calcs: dict, calculations for each feature

    Returns:
    - df: DataFrame with calculated predictors
    """

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
