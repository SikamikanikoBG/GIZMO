import numpy as np
import pandas as pd

from src.functions.printing_and_logging import print_and_log

formats = ['%Y%m', '%d%m%Y']


def convert_obj_to_date(df, object_cols):
    for el in df[object_cols].columns:
        try:
            df[el+'_date'] = df[el].astype(str)  # when date is int without conversion is wrong
            df[el+'_date'] = pd.to_datetime(df[el])
            print_and_log(f'[ OBJECT TO DATE ] Column {el} converted to {df[el+"_date"].dtypes}', 'YELLOW')
        except:
            for form in formats:
                try:
                    df[el+'_date'] = df[el].astype(str)
                    df[el+'_date'] = pd.to_datetime(df[el], format=form)
                    print_and_log(f'[ OBJECT TO DATE ] Column {el} converted to {df[el+"_date"].dtypes}', 'YELLOW')
                except:
                    pass
    return df


def calculate_date_diff_between_date_columns(df, dates_cols):
    for el in dates_cols:
        for el2 in dates_cols:
            if el != el2:
                try:
                    df[el + '_diff_days_' + el2] = abs((df[el] - df[el2]) / pd.Timedelta(days=1))
                    df[el + '_diff_hours_' + el2] = abs((df[el] - df[el2]) / pd.Timedelta(hours=1))
                    df[el + '_diff_minutes_' + el2] = abs((df[el] - df[el2]) / pd.Timedelta(minutes=1))
                    df[el + '_diff_months_' + el2] = abs(((df[el] - df[el2]) / np.timedelta64(1, 'M')))
                except:
                    pass

    return df


def extract_date_characteristics_from_date_column(df, dates_cols):
    for el in dates_cols:
        df[el + '_Week_NB'] = df[el].dt.week
        df[el + '_MonthDay_NB'] = df[el].dt.day
        df[el + '_WeekDay_NB'] = df[el].dt.dayofweek
        df[el + '_YearDay_NB'] = df[el].dt.dayofyear
    return df
