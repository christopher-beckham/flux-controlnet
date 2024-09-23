#!/bin/bash

cd ..

CHKPT=$1
if [ -z $CHKPT ]; then
  echo "Usage: bash upload_cnet.sh <experiment>/<checkpoint> <HF model name>"
  exit 1
fi

MODEL_NAME=$2
if [ -z $MODEL_NAME ]; then
  echo "Usage: bash upload_cnet.sh <experiment>/<checkpoint> <HF model name>"
  exit 1
fi

if [ -z $SAVE_DIR ]; then
  echo "SAVE_DIR not set, source env.sh?"
  exit 1
fi

shift 2

python -m scripts.upload --pretrained_cnet_checkpoint $SAVE_DIR/$CHKPT --model_name $MODEL_NAME "$@"
