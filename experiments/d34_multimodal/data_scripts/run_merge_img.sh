#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/02_merge_img_predictions.py
DATADIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/data

# Classification
echo 'Merge Image Predictions'

LABELS=( 'place_country_code' 'time_label' )
for LABEL in "${LABELS[@]}"
do
    echo $LABEL
    echo '\t -merge-dev'
    $EXEC \
        --dataset-file $DATADIR/dev/base/$LABEL.dev.tsv \
        --predictions-file $DATADIR/dev/predimg/$LABEL.csv \
        --output-file $DATADIR/dev/withimg/$LABEL.dev.tsv \
        --field-pred-id obj_uri \
        --field-data-id obj \
        --field-pred predicted_class \
        --field-target img_prediction

    echo '\t -merge-test'
    $EXEC \
        --dataset-file $DATADIR/test/base/$LABEL.test.tsv \
        --predictions-file $DATADIR/test/predimg/$LABEL.csv \
        --output-file $DATADIR/test/withimg/$LABEL.test.tsv \
        --field-pred-id obj_uri \
        --field-data-id obj \
        --field-pred predicted_class \
        --field-target img_prediction

    echo ' '
done

LABELS=( 'material_group' 'technique_group' )
for LABEL in "${LABELS[@]}"
do
    echo $LABEL
    echo '\t -merge-dev'
    $EXEC \
        --dataset-file $DATADIR/dev/base/$LABEL.dev.tsv \
        --predictions-file $DATADIR/dev/predimg/$LABEL.csv \
        --output-file $DATADIR/dev/withimg/$LABEL.dev.tsv \
        --field-pred-id obj_uri \
        --field-data-id obj \
        --field-pred predicted_class \
        --prefix-pred "http://data.silknow.org/vocabulary/facet/" \
        --field-target img_prediction

    echo '\t -merge-test'
    $EXEC \
        --dataset-file $DATADIR/test/base/$LABEL.test.tsv \
        --predictions-file $DATADIR/test/predimg/$LABEL.csv \
        --output-file $DATADIR/test/withimg/$LABEL.test.tsv \
        --field-pred-id obj_uri \
        --field-data-id obj \
        --field-pred predicted_class \
        --prefix-pred "http://data.silknow.org/vocabulary/facet/" \
        --field-target img_prediction

    echo ' '
done
