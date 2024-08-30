#!/bin/bash
cd /workspace/src
python train.py

## Make Weight Tar
cd /workspace/weights
tar -cvzf /workspace/weights.tar.gz best

echo "Tar Creation Complete. Listing Files"
tar -tvf "/workspace/weights.tar.gz"