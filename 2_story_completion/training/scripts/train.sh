#!/bin/bash
cd /workspace/src
python train.py --config_file /workspace/config.json

## Make Weight Tar
cd /workspace/weights/best
tar -cvzf /workspace/weights.tar.gz .

echo "Tar Creation Complete. Listing Files"
tar -tvf "/workspace/weights.tar.gz"