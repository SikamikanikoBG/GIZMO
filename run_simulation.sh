DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
#echo $DIR/main.py ekont
source activate gizmo
python3 -W ignore $DIR/grid_search.py --project ardi_eurchf_sell&
python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_sell&
python3 -W ignore $DIR/grid_search.py --project ardi_eurcad_buy&
python3 -W ignore $DIR/grid_search.py --project ardi_audnzd_sell&
python3 -W ignore $DIR/grid_search.py --project ardi_nzdusd_sell&



