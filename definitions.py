import os


EXTERNAL_DIR = "/srv/share/jizzmo_data"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if EXTERNAL_DIR:
    pass
else:
    EXTERNAL_DIR = os.path.dirname(os.path.abspath(__file__))


args = None
params = None

# folders and paths


#ngrok_prefix = "e482"
#api_url_prefix = "https://"+ngrok_prefix+"-95-43-20-41.eu.ngrok.io/"
api_url_prefix = "http://192.168.50.106:5000/"
api_url_post_results_models = api_url_prefix + "bulk_upload_new_raw_data"
api_url_post_results_predict = api_url_prefix + "upload_models_predict"
api_url_post_models_simulations = api_url_prefix + "upload_models_simulations"
api_url_get_input_data = api_url_prefix + "get_all_raw_data"
api_url_get_history_data = api_url_prefix + "get_history_data"
api_url_post_error = api_url_prefix + "post_error_log"

notifications_url_prefix = "https://notify.run/"
notifications_url_grid = f"{notifications_url_prefix}QcLwLsIj07xKPxH81Eys"

# GUI
selected_project = None
selected_param_file = None
input_df = None

