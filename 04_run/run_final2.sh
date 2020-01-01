#!/bin/bash

TASKS=(place timespan material technique)
TASKS=(technique)
BASEDIR=/data/euprojects/silknow
DATASET=$BASEDIR/tasks
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

echo "SINGLE (IID A)"
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

echo "FINAL (IID B)"
for task in "${TASKS[@]}"; do
    echo "$task";
    TRAIN=$DATASET/$task/final.trn.csv
    TEST=$DATASET/$task/final.tst.csv
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

echo "GROUP"
for task in "${TASKS[@]}"; do
    echo "$task";
    TRAIN=$DATASET/$task/group.trn.csv
    TEST=$DATASET/$task/group.tst.csv
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

echo "XLING A"
for task in "${TASKS[@]}"; do
    echo "$task";
    TRAIN=$DATASET/$task/xling.trn.csv
    TEST=$DATASET/$task/xling.tst.csv
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

echo "XLING B"
for task in "${TASKS[@]}"; do
    echo "$task";
    TRAIN=$DATASET/$task/xling.alt.csv
    TEST=$DATASET/$task/xling.tst.csv
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
