import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
args = None
ngrok_prefix = "a8ce"
api_url_prefix = "https://"+ngrok_prefix+"-95-43-20-41.eu.ngrok.io/"
api_url_post_results_models = api_url_prefix + "bulk_upload_new_raw_data"
api_url_post_results_predict = api_url_prefix + "upload_models_predict"
api_url_post_models_simulations = api_url_prefix + "upload_models_simulations"
api_url_get_input_data = api_url_prefix + "get_all_raw_data"
api_url_get_history_data = api_url_prefix + "get_history_data"

