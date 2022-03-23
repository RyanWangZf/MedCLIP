#! /usr/bin/env bash

python train_gcnclassifier.py \
  --name gcnclassifier_v2_ones3_t012v3t4_lr1e-6 \
  --pretrained ./weights/model_ones_3epoch_densenet.tar \
  --train-folds 012 \
  --val-folds 3 \
  --test-folds 4 \
  --lr 1e-6 \
  --batch-size 8 \
  --num-epochs 150 \
  --gpus 0 &
