import datetime

import definitions
from src.functions import api_communication

folders = ["ardi_audnzd_sell",
           "ardi_eurcad_buy",
           "ardi_eurcad_sell",
           "ardi_eurchf_sell",
           "ardi_nzdusd_sell"]

url = definitions.api_url_get_input_data

start_time = datetime.datetime.now()
in_data = api_communication.api_get(url, None)

columns_reorder = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume', 'Currency']

in_data = in_data[columns_reorder].copy()
#in_data.to_parquet(f'./input_data/ardi/daily_stock_data_month_test.parquet')
api_time = datetime.datetime.now()


for currency in in_data.Currency.unique().tolist():
    for folder in folders:
        in_data[in_data['Currency'] == currency].sort_values(by='time', ascending=True).tail(4000).to_parquet(f'./input_data/{folder}/daily_stock_data_month_{currency}.parquet')

cutting_time = datetime.datetime.now()

api_time_delta = api_time - start_time
cut_time_delta = cutting_time - api_time
total_time_delta = cutting_time - start_time

print(f"API: {api_time_delta}, Cut: {cut_time_delta}, Total: {total_time_delta}")
