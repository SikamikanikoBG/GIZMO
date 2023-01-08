import argparse
import datetime
import os
from notify_run import Notify

notify = Notify()

import definitions
from src.functions import api_communication
from src.functions.printing_and_logging import print_and_log

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
           "ardi_eurusd_buy",
           "ardi_eurusd_sell",
           "ardi_nzdusd_sell",
           "ardi_nzdusd_buy",
           "ardi_cadchf_sell",
           "ardi_cadchf_buy",
           "ardi_xauusd_buy",
           "ardi_xauusd_sell",
           "ardi_gbpusd_buy",
           "ardi_gbpusd_sell",
           "ardi_audcad_buy",
           "ardi_audcad_sell",
           "ardi_audchf_buy",
           "ardi_audchf_sell",
           "ardi_audjpy_buy",
           "ardi_audjpy_sell",
           "ardi_audusd_buy",
           "ardi_audusd_sell",
           "ardi_cadjpy_buy",
           "ardi_cadjpy_sell",
           "ardi_chfjpy_buy",
           "ardi_chfjpy_sell",
           "ardi_eurgbp_buy",
           "ardi_eurgbp_sell",
           "ardi_eurnzd_buy",
           "ardi_eurnzd_sell",
           "ardi_euraud_buy",
           "ardi_euraud_sell",
           "ardi_eurjpy_buy",
           "ardi_eurjpy_sell",
           "ardi_gbpaud_buy",
           "ardi_gbpaud_sell",
           "ardi_gbpcad_buy",
           "ardi_gbpcad_sell",
           "ardi_gbpchf_buy",
           "ardi_gbpchf_sell",
           "ardi_gbpjpy_buy",
           "ardi_gbpjpy_sell",
           "ardi_gbpnzd_buy",
           "ardi_gbpnzd_sell",
           "ardi_nzdjpy_buy",
           "ardi_nzdjpy_sell",
           "ardi_usdcad_buy",
           "ardi_usdcad_sell",
           "ardi_usdchf_buy",
           "ardi_usdchf_sell",
           "ardi_usdjpy_buy",
           "ardi_usdjpy_sell",
           "ardi_xagusd_buy",
           "ardi_xagusd_sell"
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
        print_and_log(f"[ DATA LOAD ]: the input dataframe is empty. Probably the API is not accessible...", "RED")
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
    # notify.send(f"laksjdlaksdlkjasldjas")
except Exception as e:
    print(e)
    notify.send(f"[ ArDi ERROR ] API: {e}")
