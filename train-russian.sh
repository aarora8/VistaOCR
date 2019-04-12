#!/bin/bash                                                                                                                                                                    

source ~/.bashrc

. ./path.sh
. ./cmd.sh

$cmd --gpu 1 --mem 8G data/log/train.log limit_num_gpus.sh python3 src/train_cnn_lstm_ctc.py \
        --batch-size=32 \
        --line-height=30 \
        --num_in_channels=3 \
        --num-lstm-layers=3 \
        --num-lstm-units=640 \
        --lstm-input-dim=128 \
        --lr=1e-3 \
        --datadir=data/lmdb \
        --validdirtype=validation \
        --validdir=data/lmdb \
        --snapshot-num-iterations=2000 \
        --snapshot-prefix=data/model/russian \
        --max-val-size=20000 \
        --patience 15 \
        --min-lr 1e-7 \
        --write_samples \
        --samples_dir=data/samples \
        --nepochs=250

#$cmd --gpu 1 --mem 8G data/log/test.log limit_num_gpus.sh src/decode_testset.sh data/output data/model/russian-best_model.pth data/lmdb
