DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo

python3 -W ignore $DIR/load_data.py --session predict --volumes 1000
nice -n 10 python3 -W ignore $DIR/main.py --project ardi_gbpaud_sell --predict_module standard --main_model xgb --pred_data_prep ardi  --tp 0.0025 --sl 0.015 --period 240 --nb_tree_features 15&
