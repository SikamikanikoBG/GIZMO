DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source activate gizmo
python3 -W ignore $DIR/load_data.py --session predict --volumes 1000
#nice -n 10 python3 -W ignore $DIR/main.py --project ardi_eurcad_buy --predict_module standard --main_model xgb --pred_data_prep ardi --tp 0.0025 --sl 0.008 --period 480 --nb_tree_features 30&
#nice -n 10 python3 -W ignore $DIR/main.py --project ardi_nzdusd_buy --predict_module standard --main_model xgb --pred_data_prep ardi --tp 0.0025 --sl 0.01 --period 480 --nb_tree_features 30&
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_audnzd_buy_20221117 --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.008 --period 240
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_nzdusd_buy_20221117 --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.004 --period 480
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_eurcad_buy_20221117 --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.006 --period 480