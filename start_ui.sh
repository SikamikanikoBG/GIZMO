#!/bin/bash

# run_ui.sh

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gizmo_ar

echo "Running the UI..."
python gizmo_ui.py
