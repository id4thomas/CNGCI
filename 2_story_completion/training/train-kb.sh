#!/bin/bash
## Fill in here with wandb Info
WANDB_ENTITY=""
WANDB_PROJECT="cngci-candidate-generator"
WANDB_API_KEY=""

HF_TOKEN="" ## needed for gated models

RUN_NAME="sample_kb"

CONFIG_DIR="./configs/${RUN_NAME}.json"

## RUN Training Container
docker container rm -f train-candidate-generator
docker run --gpus all \
	--name="train-candidate-generator" \
	--entrypoint /workspace/train.sh \
	-e WANDB_ENTITY=$WANDB_ENTITY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e HF_TOKEN=$HF_TOKEN \
	-v ${CONFIG_DIR}:/workspace/config.json \
	candidate-generator-trainer:cuda121 \
	/workspace/train.sh

## Copy Trained weight files
mkdir weights
docker cp train-candidate-generator:/workspace/weights.tar.gz "weights/${RUN_NAME}.tar.gz"

## Delete Container
docker container rm -f train-candidate-generator