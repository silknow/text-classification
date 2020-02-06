# SilkNOW Text Classification

## Downloading the Trained Models
Download the models:
    * [D3.3. version]()

## Requirements
See `requirements.txt` they include the following:

    * A CUDA capable GPU
    * Python and the following libraries:
        * pytorch
        * ftfy
        * sacremoses

## Command Line Tool

### Usage

```
$./sntxtclassify.py -h
usage: sntxtclassify <command> [<args>]

The available commands are:
   train        Train a text classification model
   evaluate     Evaluate a text classification model
   classify     Classify text samples

SilkNOW Text Classifier

positional arguments:
  {train,evaluate,classify}

optional arguments:
  -h, --help            show this help message and exit
```

#### Training a Model

```
./sntxtclassify.py train -h
       [-h] --data-train DATA_TRAIN --target TARGET
       [--pretrained-embeddings PRETRAINED_EMBEDDINGS] [--all-embeddings]
       --model-save MODEL_SAVE

optional arguments:
  -h, --help            show this help message and exit
  --data-train DATA_TRAIN
                        train CSV file
  --target TARGET       CSV column target
  --pretrained-embeddings PRETRAINED_EMBEDDINGS
                        pretrained embeddings path
  --all-embeddings      Keep all word vectors
  --model-save MODEL_SAVE
                        save model path

```
The train CSV file, `DATA_TRAIN` has the following requirements:

    * Tab-separated values;
    * A header;
    * `txt` field containing the text;
    * `lang` field containing the language of the text;
    * a label field with the same name as specified by the `TARGET` argument;

The pretained embeddings directory is expected to contain the fasttext
multilingual aligned embeddings available [here](https://fasttext.cc/docs/en/aligned-vectors.html).

Example:

```
./sntxtclassify.py train \
    --data-train /data/euprojects/silknow/tasks/technique/final.trn.csv \
    --target technique \
    --pretrained-embeddings /data/vectors/ftaligned \
    --model-save /data/euprojects/silknow/models/technique
```

#### Evaluating

```
./sntxtclassify.py evaluate -h
usage: sntxtclassify <command> [<args>]
       [-h] --model-load MODEL_LOAD --data-test DATA_TEST --target TARGET

optional arguments:
  -h, --help            show this help message and exit
  --model-load MODEL_LOAD
                        model path
  --data-test DATA_TEST
                        test CSV path
  --target TARGET       CSV column target
```

The test CSV file, `DATA_TEST` has the same requirements as `DATA_TRAIN`.

Example:

```
./sntxtclassify.py evaluate `
    --data-test /data/euprojects/silknow/tasks/material/final.tst.csv \
    --target material \
    --model-load /data/euprojects/silknow/models/material
```

#### Classifying Text

```
./sntxtclassify.py classify -h
usage: sntxtclassify <command> [<args>]
       [-h] [--model-load MODEL_LOAD] --data-input DATA_INPUT --data-output
       DATA_OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  --model-load MODEL_LOAD
                        model path
  --data-input DATA_INPUT
                        input CSV path
  --data-output DATA_OUTPUT
                        output CSV path

```

The `DATA_INPUT` CSV only requires a `txt` and a `lang` field.

Example:

```
./sntxtclassify.py classify \
    --data-output /tmp/ex.tsv \
    --data-input /data/euprojects/silknow/tasks/missing_material.csv  \
    --model-load /data/euprojects/silknow/models/material
```


## Library

Example:

```python
import sdnn

text = ‘’’Tapestry, woven in wool and silk, after a painting by Raphael depicting the Holy Family. Tapestry, after a painting of the Holy Family by Raphael; principal weaver Adolphe Margarita; made at the Gobelins in Paris, 23/05/1852-06/12/1856’’’
lang = ‘en’

model_path = ‘/data/silknow/models/technique’
model = sdnn.load_model(model_path)
model.classify(text, lang)
```

Alternatively, for bulk classification of multiple texts, use:

```python
model.classify_text(text_lang_pairs)
```

Where `text_lang_pairs` is a list of (text, lang) tuples.


## REST API Server

To start a local API server use the `server.py` executable. Example:

```
./server.py  --model-load /data/euprojects/silknow/models/material --port 8000
```

The server exposes a single endpoint (the root). Example request:

```python
import requests


text = ‘’’Tapestry, woven in wool and silk, after a painting by Raphael depicting the Holy Family. Tapestry, after a painting of the Holy Family by Raphael; principal weaver Adolphe Margarita; made at the Gobelins in Paris, 23/05/1852-06/12/1856’’’
lang = ‘en’

js = [{'txt': text, 'lang': lang}]

r = requests.post('http://localhost:8000', json=js)
print(r.json())
[{'label': 'silk_wool', 'score': 1.0}]
```
