DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo

touch simulations.running
nice -n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_xauusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_xauusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_cadchf_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_cadchf_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
wait
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audcad_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audcad_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audchf_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audchf_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audjpy_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audjpy_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audusd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audusd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_cadjpy_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_cadjpy_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_chfjpy_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_chfjpy_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurgbp_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurgbp_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurnzd_buy --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurnzd_sell --tp 0.0025 --sl 0.006 --training_rows 7000 --period 480 --nb_features 50&

wait
rm simulations.running



