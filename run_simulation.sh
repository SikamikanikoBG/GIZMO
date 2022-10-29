DIR="$( cd "$( dirname "$0" )" && pwd )"
# shellcheck disable=SC2164
cd $DIR
source activate gizmo
nice -n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_sell&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_buy&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_sell&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_buy&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_sell&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_buy&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_sell&
nice -n 16 python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_buy&



