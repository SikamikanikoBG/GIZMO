DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
#echo $DIR/main.py ekont
source activate gizmo
python3 -W ignore $DIR/load_data.py --session predict --volumes 1000
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_audnzd_buy --predict_module standard --main_model xgb --pred_data_prep ardi --tp 0.0025 --sl 0.004 --period 480 --nb_tree_features 50&
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_nzdusd_buy --predict_module standard --main_model xgb --pred_data_prep ardi --tp 0.0025 --sl 0.006 --period 480 --nb_tree_features 30&

