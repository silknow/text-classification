#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/02_merge_text_predictions.py

# Classification
echo 'Merge'
LABELS=( 'material_group' 'place_country_code' 'technique_group' 'time_label' )

for LABEL in "${LABELS[@]}"
do
    echo $LABEL
    echo '\t -merge-dev'
    $EXEC \
        --dataset-file ./dev/$LABEL.dev.tsv \
        --predictions-file ./dev/$LABEL.devf.tsv \
        --output-file ./dev/$LABEL.devt.tsv \
        --field-pred-id id \
        --field-data-id obj \
        --field-pred predicted \
        --field-target text_prediction

    echo '\t -merge-test'
    $EXEC \
        --dataset-file ./test/$LABEL.test.tsv \
        --predictions-file ./test/$LABEL.testf.tsv \
        --output-file ./test/$LABEL.testt.tsv \
        --field-pred-id id \
        --field-data-id obj \
        --field-pred predicted \
        --field-target text_prediction
    echo ' '
done

