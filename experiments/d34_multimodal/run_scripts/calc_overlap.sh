#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/04_calculate_overlap.py
DATADIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/data

# Classification
echo 'Classification'
LABELS=( 'material_group' 'place_country_code' 'technique_group' 'time_label' )
# DATACOLS=("img_prediction" "text_prediction")

for LABEL in "${LABELS[@]}"
do
    echo $LABEL
    echo 'dev'
    $EXEC --dataset $DATADIR/dev/img+txt/$LABEL.dev.tsv
    echo 'test'
    $EXEC --dataset $DATADIR/test/img+txt/$LABEL.test.tsv
    echo ' '
done

