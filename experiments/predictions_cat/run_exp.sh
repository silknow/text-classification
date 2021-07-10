#!/bin/bash

EXEC=./gbcls.py
TRAINDATA=./data/train_data
TESTDATA=./data/test_data
OUTPUT=./predicted
MODELDIR=./models/

./01_process.py
./02_make_train_files.py
./03_process_test_files.py


LABELS=( 'material_group' 'place_country_code' 'technique_group' 'time_label' )

for LABEL in "${LABELS[@]}"
do
    DATACOLS=("${LABELS[@]}")
    for i in "${!DATACOLS[@]}"; do
        if [[ ${DATACOLS[i]} = $LABEL ]]; then
            unset 'DATACOLS[i]'
        fi
    done
    DATACOLS+=("museum")
    echo $LABEL
    echo "${DATACOLS[@]}"
    echo 'train'
    $EXEC train \
        --data-train $TRAINDATA/$LABEL.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --model-save $MODELDIR/$LABEL
    echo 'predict'
    $EXEC classify \
        --data-input $TESTDATA/$LABEL.tsv \
        --data-output $OUTPUT/$LABEL.tsv \
        --scores \
        --cols "${DATACOLS[@]}" \
        --model-load $MODELDIR/$LABEL
    echo ' '
done
