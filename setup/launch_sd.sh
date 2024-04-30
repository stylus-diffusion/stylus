#!/bin/bash

# Check if exactly one argument is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <n>"
    exit 1
fi

# Easy way to get total gpus on machine
TOTAL_GPUS=$(nvidia-smi -L | wc -l)
SD_PER_GPU=2
MAX_SD=$((TOTAL_GPUS * SD_PER_GPU))
n=$1


# Validate that TOTAL_GPUS is a number
if ! [[ "$TOTAL_GPUS" =~ ^[0-9]+$ ]]; then
    echo "Error: Failed to determine the total number of GPUs."
    exit 1
fi

# Exit if the requested number of servers exceeds the maximum
if [ "$n" -gt $MAX_SD ]; then
    echo "Only Launch $MAX_SD Servers Max"
    exit 1
fi

# Get the directory where the script is located
STYLUSDIR=$(dirname "$(readlink -f "$0")")/..
cd "$STYLUSDIR/stable_diffusion"
port=7860

echo $STYLUSDIR

for (( i=0; i<n; i++ ))
do
    # Calculate the GPU to use
    gpu=$((i % TOTAL_GPUS))
    
    # Create log directory
    mkdir -p "$STYLUSDIR/log/"
    
    # Set CUDA_VISIBLE_DEVICES for the current process and launch it
    echo "Launching server $i on GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python launch.py --api --port=$port | tee $STYLUSDIR/log/server_$i & 
    
    # Increment the port number for the next server
    ((port++))
	sleep 2
done
