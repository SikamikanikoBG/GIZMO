import argparse
import datetime
import os
from notify_run import Notify
notify = Notify()

import definitions
from src.functions import api_communication

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--volumes', type=str, help='nb records to keep')
parser.add_argument('--session', type=str, help='to know from which api to load the data. predict or gridsearch')
args = parser.parse_args()
volumes = int(args.volumes.lower())
session = args.session.lower()

folders = ["ardi_audnzd_sell",
           "ardi_audnzd_buy",
           "ardi_eurcad_buy",
           "ardi_eurcad_sell",
           "ardi_eurchf_sell",
           "ardi_eurchf_buy",
           "ardi_nzdusd_sell",
           "ardi_nzdusd_buy",
           "ardi_cadchf_sell",
           "ardi_cadchf_buy",
            "ardi_xauusd_buy",
            "ardi_xauusd_sell",
           "ardi_gbpusd_buy",
            "ardi_gbpusd_sell"
           ]

for folder in folders:
    path = f'./input_data/{folder}'

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

if session == 'predict':
    url = definitions.api_url_get_input_data
elif session == 'gridsearch':
    url = definitions.api_url_get_history_data
else:
    url = None
    quit()

start_time = datetime.datetime.now()

try:
    in_data = api_communication.api_get(url, None)
    if in_data.empty:
        notify.send(f"ArDi ERROR: API data load: the input dataframe is empty. Probably the API is not accessible...")
        quit()

    # in_data["time"] = pd.datetime(in_data["time"])
    columns_reorder = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume', 'Currency']
    in_data = in_data[columns_reorder].copy()
    api_time = datetime.datetime.now()

    for currency in in_data.Currency.unique().tolist():
        for folder in folders:
            try:
                in_data[in_data['Currency'] == currency].sort_values(by='time', ascending=True).tail(
                    volumes).to_parquet(f'./input_data/{folder}/daily_stock_data_month_{currency}.parquet')
            except Exception as e:
                print(e)
                pass

    cutting_time = datetime.datetime.now()

    api_time_delta = api_time - start_time
    cut_time_delta = cutting_time - api_time
    total_time_delta = cutting_time - start_time

    print(f"API: {api_time_delta}, Cut: {cut_time_delta}, Total: {total_time_delta}")
except Exception as e:
    print(e)
    notify.send(f"[ ArDi ERROR ] API: {e}")
