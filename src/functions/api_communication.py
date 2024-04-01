import json

import notify_run
import requests
import pandas as pd


def api_post(url, data):
    """
    Send a POST request to the specified URL with the provided data.

    Parameters:
    - url: str, URL to send the POST request to
    - data: DataFrame, data to be sent in JSON format

    Returns:
    - response: Response object, response from the POST request
    """
    data = data.to_json(orient='records')
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    try:
        response = requests.post(url=url, data=data, headers=headers)
        return response
    except Exception as e:
        print(e)
        return e


def api_post_string(url, string):
    """
    Send a POST request to the specified URL with the provided string data.

    Parameters:
    - url: str, URL to send the POST request to
    - string: str, string data to be sent

    Returns:
    - response: Response object, response from the POST request
    """
    try:
        response = requests.post(url=f"{url}{string}")
        return response
    except Exception as e:
        print(e)
        return e


def api_get(url, conditions):
    """
    Send a GET request to the specified URL with optional conditions.

    Parameters:
    - url: str, URL to send the GET request to
    - conditions: str, additional conditions to append to the URL

    Returns:
    - df_received_data: DataFrame, data received from the API response
    """
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
