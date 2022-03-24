#! /usr/bin/env bash

# TODO (zw): need to be updated

export PYTHONPATH=$(pwd)

python /workspace/paper/train_wordsat.py --name wordsat_t401v2t3_pree147_lr1e-4_ctx --pretrained /workspace/biview/models/mlclassifier_ones3_t401v2t3_lr1e-6_e147.pth --train-folds 401 --val-folds 2 --test-folds 3 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path /datasets/vocab.pkl

python /workspace/paper/train_wordsat.py --name wordsat_t340v1t2_pree130_lr1e-4_ctx --pretrained /workspace/biview/models/mlclassifier_ones3_t340v1t2_lr1e-6_e130.pth --train-folds 340 --val-folds 1 --test-folds 2 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path /datasets/vocab.pkl

python /workspace/paper/train_wordsat.py --name wordsat_t234v0t1_pree145_lr1e-4_ctx --pretrained /workspace/biview/models/mlclassifier_ones3_t234v0t1_lr1e-6_e145.pth --train-folds 234 --val-folds 0 --test-folds 1 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path /datasets/vocab.pkl

python /workspace/paper/train_wordsat.py --name wordsat_t123v4t0_pree148_lr1e-4_ctx --pretrained /workspace/biview/models/mlclassifier_ones3_t123v4t0_lr1e-6_e148.pth --train-folds 123 --val-folds 4 --test-folds 0 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path /datasets/vocab.pkl

python /workspace/paper/train_wordsat.py --name wordsat_t012v3t4_pree150_lr1e-4_ctx --pretrained /workspace/biview/models/mlclassifier_ones3_t012v3t4_lr1e-6_e150.pth --train-folds 012 --val-folds 3 --test-folds 4 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path /datasets/vocab.pkl
