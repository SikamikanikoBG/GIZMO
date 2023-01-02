DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo

python3 -W ignore $DIR/load_data.py --session predict --volumes 1000
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_gbpnzd_sell --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.015 --period 480 --nb_tree_features 50&
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_eurnzd_sell --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.015 --period 240 --nb_tree_features 30&
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_usdchf_sell --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.015 --period 480 --nb_tree_features 30&
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_gbpchf_sell --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.015 --period 480 --nb_tree_features 50&
