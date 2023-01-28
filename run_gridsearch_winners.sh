DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo
rm -rf /srv/share/jizzmo_data/implemented_models/*

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpnzd_sell --winner yes --tp 0.0025 --sl 0.015 --training_rows 7000 --period 480 --nb_features 15
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpaud_sell --winner yes --tp 0.0025 --sl 0.015 --training_rows 4000 --period 480 --nb_features 50
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audusd_buy --winner yes --tp 0.0025 --sl 0.015 --training_rows 4000 --period 480 --nb_features 15
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpcad_sell --winner yes --tp 0.0025 --sl 0.015 --training_rows 4000 --period 480 --nb_features 15
