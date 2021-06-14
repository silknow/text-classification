#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/03_merge_img_txt.py
DATADIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/data

# Classification
echo 'Merge Image Predictions'

LABELS=( 'place_country_code' 'time_label' 'material_group' 'technique_group' )
for LABEL in "${LABELS[@]}"
do
    echo $LABEL
    echo '\t -merge-dev'
    $EXEC \
        --dataset-base $DATADIR/dev/base/$LABEL.dev.tsv \
        --dataset-img $DATADIR/dev/withimg/$LABEL.dev.tsv \
        --dataset-txt $DATADIR/dev/withtxt/$LABEL.dev.tsv \
        --output-file $DATADIR/dev/img+txt/$LABEL.dev.tsv \
        --field-id obj \
        --field-img img_prediction \
        --field-txt text_prediction

    echo '\t -merge-test'
    $EXEC \
        --dataset-base $DATADIR/test/base/$LABEL.test.tsv \
        --dataset-img $DATADIR/test/withimg/$LABEL.test.tsv \
        --dataset-txt $DATADIR/test/withtxt/$LABEL.test.tsv \
        --output-file $DATADIR/test/img+txt/$LABEL.test.tsv \
        --field-id obj \
        --field-img img_prediction \
        --field-txt text_prediction
    echo ' '
done
