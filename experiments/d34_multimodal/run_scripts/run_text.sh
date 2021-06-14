#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/repo/text-classification/sntxtclassify.py
EXP_DIR=./exp/txt-txt
MODEL_DIR=./models/text


# create embs
# echo 'Creating embeddings'
# $EXEC embeddings \
#     --data-train ./text.tsv \
#     --pretrained-embeddings /data/vectors/ftaligned/ \
#     --save $MODEL_DIR/embeddings.vec


# Classification
echo 'Classification'
LABELS=( 'material_group' 'place_country_code' 'technique_group' 'time_label' )

for LABEL in "${LABELS[@]}"
do
    echo $LABEL
    echo '\t -train'
    $EXEC train \
        --data-train train/$LABEL.train.tsv \
        --target $LABEL \
        --pretrained-embeddings $MODEL_DIR/embeddings.vec \
        --model-save $MODEL_DIR/$LABEL

    echo '\t -eval'
    $EXEC evaluate \
        --data-test ./test/$LABEL.test.tsv \
        --target $LABEL \
        --model-load $MODEL_DIR/$LABEL \
        > $EXP_DIR/$LABEL.txt

    echo '\t -classify-dev'
    $EXEC classify \
        --data-input ./dev/$LABEL.dev.tsv \
        --data-output ./dev/$LABEL.devf.tsv \
        --model-load $MODEL_DIR/$LABEL

    echo '\t -classify-test'
    $EXEC classify \
        --data-input ./test/$LABEL.test.tsv \
        --data-output ./test/$LABEL.testf.tsv \
        --model-load $MODEL_DIR/$LABEL
done

