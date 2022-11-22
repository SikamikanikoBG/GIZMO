DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo2

nice -n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
#nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_xauusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
#nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_xauusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
#nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
#nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_cadchf_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_cadchf_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&



