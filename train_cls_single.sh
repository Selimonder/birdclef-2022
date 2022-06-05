#!/bin/bash

GPU=$1
VAL_OUT_DIR=$2
CONFIG=$3
PREFIX=$4
FOLD=$5
DATA_DIR=$6

PYTHONPATH=.  python -u   train_classifier.py  --gpu $GPU \
 --config configs/${CONFIG}.json  --workers 8 --test_every 1 \
 --val-dir $VAL_OUT_DIR --folds-csv folds.csv --prefix $PREFIX  --fold $FOLD --freeze-epochs 0 --data-dir $DATA_DIR
