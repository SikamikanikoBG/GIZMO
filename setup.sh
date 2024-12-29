#!/bin/bash

# setup.sh
echo "Creating new conda environment gizmo_ar with Python 3.9..."
conda create -n gizmo_ar python=3.9 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gizmo_ar

echo "Installing pip requirements..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found in the current directory!"
    exit 1
fi

echo "Setup complete! You can now activate the environment using: conda activate gizmo_ar"
