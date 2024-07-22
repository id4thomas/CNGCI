# !/bin/bash

DATA_DIR="sample_data.tsv"
RUN_NAME="240722_1_obslm245_test"
CONFIG_DIR="generation_configs/${RUN_NAME}.json"

mkdir "results"
RESULT_DIR="results/${RUN_NAME}"
SEED=10

python run_story_completion.py \
	--data_dir $DATA_DIR \
	--config_dir $CONFIG_DIR \
	--result_dir $RESULT_DIR \
	--seed $SEED
