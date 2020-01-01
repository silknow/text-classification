#!/bin/bash
TASKS=(place)
MODELS=(TextSCNN BiLSTM BiLSTMpool HANSentence)
BASEDIR=/data/euprojects/silknow
DATASET=$BASEDIR/tasks
VOCAB=/data/euprojects/silknow/vocab.txt
EMBS=/data/euprojects/silknow/embedding_matrix.pt
BATCH_SIZE=64
EPOCHS=300
LR=1e-3
CLIP=1
RESULTS_DIR=$BASEDIR/results
RESULTS_ALL=$RESULTS_DIR/RESULTS.json
SEED=105563145

MODELS=(TextSCNN)
HIDDEN_SIZE=0
FILTERS=300
DROPOUT=0.2
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

MODELS=(BiLSTMpool)
HIDDEN_SIZE=300
SEQLEN=400
DROPOUT=0.3
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
            --seqlen $SEQLEN --hidden_size $HIDDEN_SIZE --dropout $DROPOUT \
            --seed $SEED --results-append $RESULTS_ALL \
            --results $RESULTS
    done
done


MODELS=(HANSentence)
HIDDEN_SIZE=500
SEQLEN=300
DROPOUT=0.3
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
            --seqlen $SEQLEN --hidden_size $HIDDEN_SIZE --dropout $DROPOUT \
            --seed $SEED --results-append $RESULTS_ALL \
            --results $RESULTS
    done
done
