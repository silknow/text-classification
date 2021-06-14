#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/gbcls.py
DATADIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/data
EXPDIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/exp/cat
MODELDIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/models/cat


# Classification
echo 'TRAIN=TRAIN'
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
    DATACOLS+=("museum")
    echo $LABEL
    echo "${DATACOLS[@]}"
    echo 'train'
    $EXEC train \
        --data-train $DATADIR/train/$LABEL.train.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --model-save $MODELDIR/$LABEL
    echo 'eval'
    $EXEC evaluate \
        --data-test $DATADIR/test/base/$LABEL.test.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --model-load $MODELDIR/$LABEL \
        > $EXPDIR/$LABEL.txt
    echo ' '
done


echo 'TRAIN=DEV'
EXPDIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/exp/catd
MODELDIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/models/catd
echo 'Classification'

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
        --data-train $DATADIR/dev/base/$LABEL.dev.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --model-save $MODELDIR/$LABEL
    echo 'eval'
    $EXEC evaluate \
        --data-test $DATADIR/test/base/$LABEL.test.tsv \
        --cols "${DATACOLS[@]}" \
        --target $LABEL \
        --model-load $MODELDIR/$LABEL \
        > $EXPDIR/$LABEL.txt
    echo ' '
done

