#!/bin/bash
export WANDB_PROJECT="CNGCI"
export WANDB_ENTITY="id4thomas"
export WANDB_API_KEY=""

CONFIG_DIR="${PWD}/configs/obstacle_generator.json"

## Check directories
create_directory() {
    local directory=$1
    # Check if the directory exists
    if [ -d "$directory" ]; then
        echo "Directory $directory already exists."
    else
        echo "Directory $directory does not exist. creating"
        mkdir -p "$directory"
        if [ $? -eq 0 ]; then
            echo "Directory created."
        else
            echo "Failed to create directory."
        fi
    fi
}

create_directory "logs"
create_directory "weights"

cd train
python train_obstacle_generator.py --config_dir $CONFIG_DIR