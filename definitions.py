import os


EXTERNAL_DIR = ""
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if EXTERNAL_DIR:
    pass
else:
    EXTERNAL_DIR = os.path.dirname(os.path.abspath(__file__))


args = None
params = None

# folders and paths

# GUI
selected_project = None
selected_param_file = None
input_df = None

# XGBoost
n_estimators = 100
early_stopping_rounds = 10
learning_rate = 0.01
max_features = 30

# MLFlow 
mlflow_tracking_uri = "http://10.128.11.44:8503"

mlflow_prefix = "Scoring Gizmo"
