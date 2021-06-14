#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/gbcls.py
DATADIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/data
EXPDIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/exp/cat+img
MODELDIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/models/cat+img

# Classification
echo 'Classification'
LABELS=( 'material_group' 'place_country_code' 'technique_group' 'time_label' )

for LABEL in "${LABELS[@]}"
do
    DATACOLS=("${LABELS[@]}")
    for i in "${!DATACOLS[@]}"; do
        if [[ ${DATACOLS[i]} = $LABEL ]]; then
            unset 'DATACOLS[i]'
        fi
    done
    DATACOLS+=("img_prediction")
    echo $LABEL
    echo "${DATACOLS[@]}"
    echo 'train'
    $EXEC train \
        --data-train $DATADIR/dev/withimg/$LABEL.dev.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --model-save $MODELDIR/$LABEL
    echo 'eval'
    $EXEC evaluate \
        --data-test $DATADIR/test/withimg/$LABEL.test.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --model-load $MODELDIR/$LABEL \
        > $EXPDIR/$LABEL.txt
    echo ' '
done

