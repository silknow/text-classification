#!/bin/bash

TASKS=(place timespan technique material)
BASEDIR=/data/euprojects/silknow
DATASET=$BASEDIR/tasks
VOCAB=/data/euprojects/silknow/vocab.txt
EMBS=/data/euprojects/silknow/embedding_matrix.pt
BATCH_SIZE=64
EPOCHS=300
LR=3e-4
CLIP=0
RESULTS_DIR=$BASEDIR/results
RESULTS_ALL=$RESULTS_DIR/RESULTS.json
SEED=105563145

MODELS=(TextGCNN)
FILTERS=200
HIDDEN_SIZE=0
DROPOUT=0.3
SEQLEN=300
echo "$RESULTS_ALL"
echo "IID"
for task in "${TASKS[@]}"; do
    echo "$task";
    TRAIN=$DATASET/$task/iid.trn.csv
    TEST=$DATASET/$task/iid.tst.csv
    for model in "${MODELS[@]}"; do
        RESULTS=$RESULTS_DIR/$task.$model.iid.txt
        ./dnn.py $TRAIN $TEST $VOCAB $EMBS --col-txt txt --col-tgt $task \
            --model $model \
            --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --clip $CLIP \
            --seqlen $SEQLEN \
            --filters $FILTERS --hidden_size $HIDDEN_SIZE --dropout $DROPOUT \
            --seed $SEED --results-append $RESULTS_ALL \
            --results $RESULTS
    done
done
