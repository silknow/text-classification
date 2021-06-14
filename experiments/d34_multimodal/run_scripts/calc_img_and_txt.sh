#!/bin/bash

EXEC=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/05_results_of_image_and_text_classifiers.py
DATADIR=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/data/test/img+txt
EXPDIR_IMG=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/exp/img
EXPDIR_TXT=~/Dropbox/Work/EUProjects/silknow/jointtextmodule2/exp/txt


LABELS=( 'material_group' 'place_country_code' 'technique_group' 'time_label' )

for LABEL in "${LABELS[@]}"
do
    echo 'txt'
    $EXEC --dataset $DATADIR/$LABEL.test.tsv \
        --field-target $LABEL \
        --field-pred "text_prediction" \
        > $EXPDIR_TXT/$LABEL.txt
    echo ' '

    echo 'img'
    $EXEC --dataset $DATADIR/$LABEL.test.tsv \
        --field-target $LABEL \
        --field-pred "img_prediction" \
        > $EXPDIR_IMG/$LABEL.txt
    echo ' '
done
