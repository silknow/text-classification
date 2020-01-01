#!/bin/bash

TASKS=(place timespan material technique)
BASEDIR=/data/euprojects/silknow
DATASET=$BASEDIR/tasks
MODELDIR=$BASEDIR/models
VOCAB=/data/euprojects/silknow/vocab.txt
EMBS=/data/euprojects/silknow/embedding_matrix.pt
BATCH_SIZE=64
EPOCHS=300
LR=0.005
CLIP=0
RESULTS_DIR=$BASEDIR/results
RESULTS_ALL=$RESULTS_DIR/RESULTS_FINAL.json
SEED=105563145

MODELS=(TextGCNN)
FILTERS=100
HIDDEN_SIZE=0
DROPOUT=0.4
SEQLEN=300
echo "$RESULTS_ALL"


echo "FINAL (IID B)"
for task in "${TASKS[@]}"; do
    echo "$task";
    TRAIN=$DATASET/$task/final.trn.csv
    TEST=$DATASET/$task/final.tst.csv
    for model in "${MODELS[@]}"; do
        RESULTS=$RESULTS_DIR/$task.$model.iid.txt
        ./dnn.py $TRAIN $TEST $VOCAB $EMBS --col-txt txt --col-tgt $task \
            --model $model \
            --model-save $MODELDIR/$task \
            --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --clip $CLIP \
            --seqlen $SEQLEN \
            --filters $FILTERS --hidden_size $HIDDEN_SIZE --dropout $DROPOUT \
            --seed $SEED --results-append $RESULTS_ALL \
            --results $RESULTS
    done
done
