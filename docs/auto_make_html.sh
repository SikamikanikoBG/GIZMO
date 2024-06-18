#!/bin/bash

root_dir = docs/

#if [ ! -d "$root_dir/logs" ]; then
#
#  chmod -w "$root_dir"
#
#  echo "No log folder detected. Creating one.."
#  # Create the logs directory if it doesn't exist
#  mkdir -p "$root_dir/logs" || {
#    echo "Error: Failed to create logs directory. Please check permissions."
#    exit 1  # Exit the script with an error code
#  }
#fi

make html 2>&1 | tee ./logs/log_$(date +"%Y-%m-%d_%H-%M-%S").txt
