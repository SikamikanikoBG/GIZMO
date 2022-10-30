DIR="$( cd "$( dirname "$0" )" && pwd )"
# shellcheck disable=SC2164
cd $DIR
source activate gizmo
#nice 0-n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_sell --winner yes --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_buy --winner yes --tp 0.0025 --sl 0.01 --training_rows 10000 --period 480 --nb_features 30&

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_sell --winner yes --tp 0.0025 --sl 0.004 --training_rows 4000 --period 120 --nb_features 50&

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_buy --winner yes --tp 0.0025 --sl 0.01 --training_rows 10000 --period 480 --nb_features 50&

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_buy --winner yes --tp 0.0025 --sl 0.01 --training_rows 4000 --period 480 --nb_features 50&

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_sell --winner yes --tp 0.004 --sl 0.004 --training_rows 10000 --period 480 --nb_features 30&

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_sell --winner yes --tp 0.004 --sl 0.006 --training_rows 4000 --period 240 --nb_features 30&




