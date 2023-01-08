DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

# activate anaconda env
source /home/ardi/anaconda3/etc/profile.d/conda.sh
conda activate jizzmo

# create temp file as flag
touch simulations.running

# Load the data for the simulations
nice -n 16 python3 -W ignore $DIR/load_data.py --session gridsearch --volumes 15000

# Run the simulations described in run_s_parallel... Each line has couple of values to be passed as py args divided by ' '
# TRY -j 0 - this will run as many jobs as possible
# --load 100% - specify % of system load
#--memfree 1G check if there is a free memory

cat $DIR/run_simulation_parallel|/home/ardi/apps/parallel/bin/parallel --progress --bar --load 80% --memfree 1G --colsep ' ' -j 30 nice -n 16 python3 -W ignore $DIR/grid_search.py  --project {1} --tp {2} --sl {3} --training_rows {4} --period {5} --nb_features 50 

# Wait until previous command  finishes before proceeding further
wait

# delete the temp flag file
rm simulations.running

# notify for ready simulations
curl https://notify.run/QcLwLsIj07xKPxH81Eys -d "Simulations are ready! Go and check them on www.snastroenie.com!"

# Remove all unnecessary files
#rm -rf $DIR/input_data/*
#rm -rf $DIR/output_data/*
