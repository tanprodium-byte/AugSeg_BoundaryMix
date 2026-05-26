#!/usr/bin/env bash
set -euo pipefail

tport=${MASTER_PORT:-53931}
ngpu=${NPROC_PER_NODE:-1}
ROOT=.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
torchrun --standalone --nproc_per_node=${ngpu} --master_port=${tport} \
  $ROOT/train_semi.py \
  --config=$ROOT/exps/boundary_mix_v1/voc_semi662/a0_baseline/config.yaml \
  --seed ${SEED:-2} --port ${tport} "$@"
