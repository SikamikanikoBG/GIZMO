import pandas as pd

import definitions
from src import print_and_log


def missing_values(df, missing_treatment, input_data_project_folder):
    """
    Handle missing values in a DataFrame based on specified treatment.

    Steps:
    1. Calculate the percentage of missing values in each column.
    2. Create a DataFrame with column names and their respective missing value percentages.
    3. Save the DataFrame to a CSV file in the output directory.
    4. Drop columns with missing values exceeding 50% (commented out).
    5. Drop rows with missing values.
    6. Apply the specified missing treatment:
        - 'delete': Drop rows with missing values.
        - 'column_mean': Fill missing values with column means.
        - 'median': Fill missing values with column medians.
        - Other: Fill missing values with the specified treatment.
            - If any missing values remain, handle them based on data type:
                - Integer, float, uint, uint8: Fill with 0.
                - Datetime: Fill with '1900-01-01'.
                - Bool: Fill with True.
                - Object: Fill with 'MissingInformation'.
                - Log a warning for unhandled columns.
    7. Calculate the number of rows removed due to missing values and the percentage removed.

    Parameters:
        df: DataFrame, input DataFrame
        missing_treatment: str, method to handle missing values
        input_data_project_folder: str, folder path for output data

    Returns:
        df: DataFrame with missing values treated
    """

    # logging

    # removing columns with 50% or more missing values
    print_and_log('[ MISSING ] Treating missing values as specified in the param file ', 'GREEN')
    # missing_cols = df[df.columns[df.isnull().mean() > 0.99]].columns.to_list()
    # print_and_log(f'[ MISSING ] The following columns are with > 70 perc of missing values '
    #              f'and will be deleted: {missing_cols}', '')
    percent_missing = df.isna().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df.to_csv(definitions.ROOT_DIR + '/output_data/' + input_data_project_folder + '/missing_values.csv',
                            index=False)
    #df = df.drop(columns=missing_cols)

    # drop rows with na
    len_before = len(df)

    if missing_treatment == 'delete':
        df = df.dropna()
    elif missing_treatment == 'column_mean':
        column_means = df.mean()
        df = df.fillna(column_means)
        df = df.dropna()
    elif missing_treatment == 'median':
        column_median = df.median()
        df = df.fillna(column_median)
        df = df.dropna()
    else:
        df = df.fillna(missing_treatment)
        if df.isnull().values.any():
            print_and_log(' WARNING: Not all missing were treated! The following missing in the columns will be '
                          'filled with: 0 for numerical, 1900-01-01 for dates, True for bool, else MissingInformation string',
                          'YELLOW')

            for f in df.columns:

                # integer
                if df[f].dtype == "int":
                    df[f] = df[f].fillna(0)
                elif df[f].dtype == "float":
                    df[f] = df[f].fillna(0)
                elif df[f].dtype == "uint":
                    df[f] = df[f].fillna(0)
                elif df[f].dtype == "uint8":
                    df[f] = df[f].fillna(0)
                elif df[f].dtype == '<M8[ns]':
                    df[f] = df[f].fillna(pd.to_datetime('1900-01-01'))

                elif df[f].dtype == 'datetime64[ns]':
                    df[f] = df[f].fillna(pd.to_datetime('1900-01-01'))
                elif df[f].dtype == 'bool':
                    df[f] = df[f].fillna(True)
                elif df[f].dtype == object:
                    df[f] = df[f].fillna('MissingInformation')
                else:
                    print_and_log(f"[ MISSING ] Column {f} was not treated. Dtype is {df[f].dtype}", "YELLOW")
                    pass

    len_after = len(df)
    removed_missing_rows = len_before - len_after
    print_and_log(f'MISSING: rows removed due to missing: '
                  f'{removed_missing_rows} ({round(removed_missing_rows / len_before, 2)}', '')

    return df
