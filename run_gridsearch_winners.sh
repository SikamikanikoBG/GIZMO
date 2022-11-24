DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo

nice -n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurusd_buy --winner yes --tp 0.0025 --sl 0.008 --training_rows 10000 --period 480 --nb_features 50

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_buy --winner yes --tp 0.004 --sl 0.008 --training_rows 7000 --period 240 --nb_features 50

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpusd_buy --winner yes --tp 0.0025 --sl 0.004 --training_rows 10000 --period 480 --nb_features 50

















