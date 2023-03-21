DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo

# notify for ready simulations
curl https://notify.run/QcLwLsIj07xKPxH81Eys -d "GIZMO: GRID restarted! Starting API after the restart..."

python api.py
