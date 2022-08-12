import pandas as pd

from src import print_and_log


def missing_values(df, missing_treatment, input_data_project_folder):
    # logging

    # removing columns with 50% or more missing values
    print_and_log('[ MISSING ] Treating missing values as specified in the param file ', 'GREEN')
    missing_cols = df[df.columns[df.isnull().mean() > 0.70]].columns.to_list()
    print_and_log(f'[ MISSING ] The following columns are with > 70 perc of missing values '
                  f'and will be deleted: {missing_cols}', '')
    percent_missing = df.isna().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df.to_csv('./output_data/' + input_data_project_folder + '/missing_values.csv', index=False)
    df = df.drop(columns=missing_cols)

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
                    # print(f'{f} filled with 0')
                    #print_and_log(f'[ MISSING ] {f} filled with 0', '')

                elif df[f].dtype == "float":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    #print_and_log(f'[ MISSING ] {f} filled with 0', '')

                elif df[f].dtype == "uint":
                    df[f] = df[f].fillna(0)
                    # print(f'{f} filled with 0')
                    #print_and_log(f'[ MISSING ] {f} filled with 0', '')

                # dates
                elif df[f].dtype == '<M8[ns]':
                    df[f] = df[f].fillna(pd.to_datetime('1900-01-01'))
                    # print(f'{f} filled with 1900-01-01')
                    #print_and_log(f'[ MISSING ] {f} filled with 1900-01-01', '')

                # boolean
                elif df[f].dtype == 'bool':
                    df[f] = df[f].fillna(True)
                    # print(f'{f} filled with True')
                    #print_and_log(f'[ MISSING ] {f} filled with True', '')

                # string
                else:
                    df[f] = df[f].fillna('MissingInformation')
                    # print(f'{f} filled with MissingInformation')
                    #print_and_log(f'[ MISSING ] {f} filled with MissingInformation', '')

    len_after = len(df)
    removed_missing_rows = len_before - len_after
    print_and_log(f'MISSING: rows removed due to missing: '
                  f'{removed_missing_rows} ({round(removed_missing_rows / len_before, 2)}', '')

    return df