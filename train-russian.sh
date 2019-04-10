#!/bin/bash                                                                                                                                                                    

source ~/.bashrc

. ./path.sh
. ./cmd.sh

$cmd --gpu 1 --mem 8G local/train.log limit_num_gpus.sh python3 src/train_cnn_lstm_ctc.py \
        --batch-size=32 \
        --line-height=30 \
        --num_in_channels=3 \
        --num-lstm-layers=3 \
        --num-lstm-units=640 \
        --lstm-input-dim=128 \
        --lr=1e-3 \
        --datadir=local \
        --validdirtype=validation \
        --validdir=local \
        --snapshot-num-iterations=2000 \
        --snapshot-prefix=/export/b04/aarora8/aavista2/local/russian \
        --max-val-size=20000 \
        --patience 15 \
        --min-lr 1e-7 \
        --write_samples \
        --samples_dir=/export/b04/aarora8/aavista2/local \
        --nepochs=250

$cmd --gpu 1 --mem 8G local/test.log limit_num_gpus.sh src/decode_testset.sh /export/b04/aarora8/aavista2/local/ /export/b04/aarora8/aavista2/local/russian-best_model.pth /export/b04/aarora8/aavista2/local/
