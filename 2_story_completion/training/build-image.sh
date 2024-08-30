#!/bin/bash
docker build -t candidate-generator-base:cuda121 -f Dockerfile-base.cuda12 .
docker build -t candidate-generator-trainer:cuda121 -f Dockerfile-trainer.cuda12 .