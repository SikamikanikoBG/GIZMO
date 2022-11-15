import json

import notify_run
import requests
import pandas as pd


def api_post(url, data):
    data = data.to_json(orient='records')
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    try:
        response = requests.post(url=url, data=data, headers=headers)
        return response
    except Exception as e:
        print(e)
        return e




def api_get(url, conditions):
    if conditions:
        url = url + conditions
    response = requests.get(url)
    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        notification = notify_run.Notify()
        notification.send(f"[ ArDi ERROR ] API: Response status code: {response.status_code}")
        df_received_data = pd.DataFrame()
    else:
        df_received_data = pd.read_json(response.content)
    return df_received_data
