#!/bin/bash

EXE=/home/rei/Dropbox/Work/EUProjects/silknow/repo/text-classification/sntxtclassify.py
TXT=./data/text.tsv
EMBEDDINGS=/data/vectors/ftaligned/
MODEL_DIR=./models
TRAIN_DIR=./data/train_data
TEST_DIR=./data/test_data
RES_DIR=./predicted

# Extract and process data provided from the KG dump.
# This is expected to be in ./data/total_post.csv
# It creates 2 files: data/processed.tsv and data/text_train.tsv
./01_process.py

# Fix test data
# Processes the test data exported from the KG.
# This is expected to be under ./data/test_data
# It will store TSV files in the same directory as the resulting test files.
./02_fix_test.py

# Extract text
# Extracts the text from the test files and combines it with the text from
# the train files in ./data/text.tsv
./03_extract_all_text.py

# Make separate train files per task
# Splits ./data/processed.tsv into separate task specific train files stored in
# ./data/train_data/
./04_make_train.py

# Create Embeddings
# Creates an embeddings file containing all the voculary found in
# ./data/text.tsv - this makes the model size manageable instead of including
# all word embeddings for all languages in each model.
run_command="$EXE embeddings --data-train $TXT --pretrained-embeddings $EMBEDDINGS --save $MODEL_DIR/embeddings.vec"
echo 'Creating embeddings'
echo "$run_command"
eval "$run_command"

# Train Classification
# This creates the text classification models for each task using the text
# classifier and the selected embeddings. Stores them in $MODEL_DIR
TARGETS="technique_group place_country_code material_group time_label"

for target in $TARGETS; do
    echo "TRAIN: $target"
    run_command="$EXE train --data-train $TRAIN_DIR/${target}.tsv --target ${target} --pretrained-embeddings ${MODEL_DIR}/embeddings.vec --model-save ${MODEL_DIR}/${target}"
    echo "$run_command"
    eval "$run_command"
done


# Classify Test Data
# This uses the text classifier and the previously trained models to classify
# the data under ./data/test_data/*.tsv
TARGETS=('technique_group' 'place_country_code' 'material_group' 'time_label')
FILES=('technique' 'place' 'material' 'time-span')

for i in "${!TARGETS[@]}"; do
    printf '%s: %s -> %s\n' "$i" "${FILES[i]}"  "${TARGETS[i]} -> ${RES_DIR}/${TARGETS[i]}.tsv"
    run_command="${EXE} classify --data-input ${TEST_DIR}/${FILES[i]}.tsv --model-load ${MODEL_DIR}/${TARGETS[i]} --data-output ${RES_DIR}/${FILES[i]}.tsv --scores"
    eval "$run_command"
done

