DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

# activate anaconda env
source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo

# create temp file as flag
touch simulations.running

# Load the data for the simulations
nice -n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000

sh $DIR/run_gridsearch_winners.sh

curl https://notify.run/QcLwLsIj07xKPxH81Eys -d "Winners are refreshed. Cheers!"

# delete the temp flag file
rm simulations.running

