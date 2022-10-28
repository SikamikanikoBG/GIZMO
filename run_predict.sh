DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
#echo $DIR/main.py ekont
source activate gizmo
python3 -W ignore $DIR/load_data.py --session predict --volumes 1000
# python3 -W ignore $DIR/main.py --project ardi_eurchf_sell --predict_module standard --main_model xgb --pred_data_prep ardi --tp 0.0025 --sl 0.0080 --period 480 --nb_tree_features 30
python3 -W ignore $DIR/main.py --project ardi_audnzd_sell --predict_module standard --main_model xgb --pred_data_prep ardi --tp 0.0025 --sl 0.0040 --period 480 --nb_tree_features 30&
python3 -W ignore $DIR/main.py --project ardi_eurcad_buy --predict_module standard --main_model xgb --pred_data_prep ardi --tp 0.0040 --sl 0.0060 --period 480 --nb_tree_features 30



