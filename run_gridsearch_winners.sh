DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo
rm -rf /srv/share/jizzmo_data/implemented_models/*

nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpaud_buy --winner yes --tp 0.0025 --sl 0.015 --training_rows 6000 --period 480 --nb_features 30
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_gbpnzd_buy --winner yes --tp 0.0025 --sl 0.015 --training_rows 10000 --period 480 --nb_features 30
