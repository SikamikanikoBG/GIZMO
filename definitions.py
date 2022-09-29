import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
args = None
ngrok_prefix = "20aa"
api_url_prefix = "https://"+ngrok_prefix+"-78-90-241-81.eu.ngrok.io/"
api_url_post_results_models = api_url_prefix + "bulk_upload_new_raw_data"
api_url_post_results_predict = api_url_prefix + "bulk_upload_new_raw_data"
api_url_post_models_simulations = api_url_prefix + "bulk_upload_models_simulations"
api_url_get_input_data = api_url_prefix + "get_all_raw_data"

