import numpy as np
from stockstats import StockDataFrame as Sdf
import definitions
from src.functions.printing_and_logging import print_and_log


def run(df):
    # df = df[df["Direct_Active"] > 0].copy()
    print(f"[ CUSTOM CALCULATIONS ] Creating subset. New nb of rows in the table {len(df)}")
    return df


def calculate_criterion(df, predict_module):
    # df['criterion_NbDirectApp_2M'] = np.where((df['NbDirectApp_2M'] > 0), 1, 0)
    return df
