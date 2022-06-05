#!/bin/bash

NUM_GPUS=$1
VAL_OUT_DIR=$2
CONFIG=$3
FOLD=$4
DATA_DIR=$5
FOLDS_CSV=$6
PREFIX=$7

PYTHONPATH=.  python -u -m torch.distributed.launch  --nproc_per_node=$NUM_GPUS  --master_port 9979  train_classifier.py  \
 --world-size $NUM_GPUS   --distributed  --config configs/${CONFIG}.json  --workers 12 --test_every 1 \
 --val-dir $VAL_OUT_DIR --folds-csv $FOLDS_CSV --prefix $PREFIX  --fold $FOLD --freeze-epochs 0 --fp16 --data-dir $DATA_DIR
