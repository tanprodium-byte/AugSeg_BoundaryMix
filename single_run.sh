#!/usr/bin/env bash
set -e

tport=53907
ngpu=1
ROOT=.

export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nproc_per_node=${ngpu} --master_port=${tport} \
  $ROOT/train_semi.py \
  --config=$ROOT/exps/zrun_vocs_u2pl/voc_semi662/config_semi.yaml \
  --seed 2 --port ${tport}
