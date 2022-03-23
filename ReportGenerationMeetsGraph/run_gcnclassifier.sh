#! /usr/bin/env bash

python /workspace/biview/train_gcnclassifier.py --name gcnclassifier_v1_ones3_t401v2t3_lr1e-6 --pretrained /workspace/biview/models/pretrained/model_ones_3epoch_densenet.tar --dataset-dir /workspace/biview --train-folds 401 --val-folds 2 --test-folds 3 --lr 1e-6 --batch-size 8 --gpus 0 --num-epochs 150
