DIR="$( cd "$( dirname "$0" )" && pwd )"

cd $DIR
source activate gizmo
nice -n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_buy_20221117 --winner yes --tp 0.0025 --sl 0.008 --training_rows 4000 --period 240 --nb_features 50
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_buy_20221117 --winner yes --tp 0.0025 --sl 0.004 --training_rows 4000 --period 480 --nb_features 50
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_buy_20221117 --winner yes --tp 0.0025 --sl 0.006 --training_rows 4000 --period 480 --nb_features 50













