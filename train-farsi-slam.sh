#!/bin/bash                                                                                                                                                                    

source ~/.bashrc

. ./path.sh
. ./cmd.sh

mkdir -p slam_farsi

$cmd --gpu 1 --mem 8G slam_farsi/train.log limit_num_gpus.sh python3 src/train_cnn_lstm.py \
        --batch-size=32 \
        --line-height=30 \
        --num_in_channels=3 \
        --rtl \
        --num-lstm-layers=3 \
        --num-lstm-units=640 \
        --lstm-input-dim=64 \
        --lr=1e-3 \
        --datadir=slam_farsi_traindevtest \
        --snapshot-num-iterations=405 \
        --snapshot-prefix=/export/b04/aarora8/aavista/slam_farsi/fa_slam \
        --patience 20 \
        --min-lr 1e-5


$cmd --gpu 1 --mem 8G slam_farsi/train.log limit_num_gpus.sh python3 src/train_cnn_lstm_ctc.py \
        --batch-size=32 \
        --line-height=30 \
        --num_in_channels=3 \
        --num-lstm-layers=3 \
        --num-lstm-units=640 \
        --lstm-input-dim=128 \
        --lr=1e-3 \
        --datadir=${TMPDIR}/lmdb \
        --validdirtype=validation \
        --validdir=${TMPDIR}/lmdb \
        --snapshot-num-iterations=2000 \
        --snapshot-prefix=$EXPDIR/model/ckpt \
        --max-val-size=20000 \
        --patience 15 \
        --augment=$AUGMENT \
        --min-lr 1e-7 \
        --write_samples \
        --samples_dir=$EXPDIR/samples \
        --nepochs=250

$cmd --gpu 1 --mem 8G slam_farsi/test.log limit_num_gpus.sh src/decode_testset.sh slam_farsi/fa_slam-best_model.pth slam_farsi_traindevtest
