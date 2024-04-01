import pandas as pd

from src.functions.printing_and_logging import print_and_log

formats = ['%Y-%m', '%Y%m', '%d%m%Y', '%Y%m%d', '%Y-%m-%d']


def convert_obj_to_date(df, object_cols, suffix):
    """
    Convert object columns in a DataFrame to datetime format.

    Parameters:
    - df: DataFrame
    - object_cols: list of str, columns to convert to datetime
    - suffix: str, suffix to add to new datetime columns

    Returns:
    - df: DataFrame with object columns converted to datetime
    """
    switched = []
    for el in object_cols:
        try:
            df[el+suffix] = df[el].astype(str)  # when date is int without conversion is wrong
            df[el+suffix] = pd.to_datetime(df[el])
            switched.append(el)
        except:
            for form in formats:
                try:
                    df[el+suffix] = df[el].astype(str)
                    df[el+suffix] = pd.to_datetime(df[el], format=form)
                    switched.append(el)
                except:
                    pass
            del df[el+suffix]    
    print_and_log(f'[ OBJECT TO DATE ] Column converted to dates: {len(switched)}', '')
    return df


def calculate_date_diff_between_date_columns(df, dates_cols):
    """
    Calculate the difference in hours between date columns in a DataFrame.

    Parameters:
    - df: DataFrame
    - dates_cols: list of str, columns containing dates

    Returns:
    - df: DataFrame with new columns showing the difference in hours between date columns
    """
    for el in dates_cols:
        for el2 in dates_cols:
            if el != el2:
                try:
                    #df[el + '_diff_days_' + el2] = abs((df[el] - df[el2]) / pd.Timedelta(days=1))
                    df[el + '_diff_hours_' + el2] = abs((df[el] - df[el2]) / pd.Timedelta(hours=1))
                    #df[el + '_diff_minutes_' + el2] = abs((df[el] - df[el2]) / pd.Timedelta(minutes=1))
                    #df[el + '_diff_months_' + el2] = abs(((df[el] - df[el2]) / np.timedelta64(1, 'M')))

                    #if len(df[el + '_diff_days_' + el2]) == 1: del df[el + '_diff_days_' + el2]
                    if len(df[el + '_diff_hours_' + el2]) == 1: del df[el + '_diff_hours_' + el2]
                    #if len(df[el + '_diff_minutes_' + el2]) == 1: del df[el + '_diff_minutes_' + el2]
                    #if len(df[el + '_diff_months_' + el2]) == 1: del df[el + '_diff_months_' + el2]
                except:
                    pass

    return df


def extract_date_characteristics_from_date_column(df, dates_cols):
    """
    Extract specific date characteristics from date columns in a DataFrame.

    Parameters:
    - df: DataFrame
    - dates_cols: list of str, columns containing dates

    Returns:
    - df: DataFrame with new columns showing extracted date characteristics
    """
    for el in dates_cols:
        #df[el + '_Week_NB'] = df[el].dt.week
        df[el + '_MonthDay_NB'] = df[el].dt.day
        df[el + '_WeekDay_NB'] = df[el].dt.dayofweek
        #df[el + '_YearDay_NB'] = df[el].dt.dayofyear

        #if len(df[el + '_Week_NB']) == 1: del df[el + '_Week_NB']
        if len(df[el + '_MonthDay_NB']) == 1: del df[el + '_MonthDay_NB']
        if len(df[el + '_WeekDay_NB']) == 1: del df[el + '_WeekDay_NB']
        #if len(df[el + '_YearDay_NB']) == 1: del df[el + '_YearDay_NB']

    return df
