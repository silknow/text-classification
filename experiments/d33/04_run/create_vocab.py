#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

import csv
from vocabulary import VocabMultiLingual
from vechelper import load_vector_words
from nltk.corpus import stopwords
import unicodedata

# DATA_FILE = '../dataset/dataset.prp.csv'
# TEXT = 'text'
DATA_FILE = '/data/euprojects/silknow/dataset.prp.csv'
TEXT = 'txt'

LANG = 'lang'
# VOCAB_SAVE = '../dataset/vocab.txt'
VOCAB_SAVE = '/data/euprojects/silknow/vocab.txt'
VOCAB_MISSING = '/data/euprojects/silknow/temp/missing.txt'
VOCAB_ENGLISH = '/data/euprojects/silknow/temp/english.txt'

VECTORS = {
    'en': '/data/vectors/ftaligned/wiki.en.align.vec',
    'ca': '/data/vectors/ftaligned/wiki.ca.align.vec',
    'es': '/data/vectors/ftaligned/wiki.es.align.vec',
    'fr': '/data/vectors/ftaligned/wiki.fr.align.vec',
}

SWS = {
    'en': set(stopwords.words('english')),
    'es': set(stopwords.words('spanish')),
    'fr': set(stopwords.words('french')),
    'ca': set(),
}


# vectors
print('Loading vector words')
vector_words = {}
for lang in VECTORS:
    vector_words[lang] = set(load_vector_words(VECTORS[lang]))

# Vocabularies from data
print('Creating vocab')
vocab = VocabMultiLingual(sos=None, eos=None, unk=None)
missing_vocab = VocabMultiLingual(sos=None, eos=None, unk=None)
english_vocab = VocabMultiLingual(sos=None, eos=None, unk=None)

line_count = 0
with open(DATA_FILE) as src_file:
    reader = csv.DictReader(src_file, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        text = row[TEXT]
        lang = row[LANG]

        line_count += 1
        if line_count % 1000 == 0:
            print(line_count)
        text = unicodedata.normalize('NFKC', text)

        seq = vocab.sentence_to_multilingual_sequence(text, lang)

        # remove stopwords
        """
        seq = [(lang, word) for lang, word in seq
               if word not in SWS[lang]]
        """

        # found in vectors
        fnd = [(lang, word) for lang, word in seq
               if word in vector_words[lang]]
        vocab.add_sequence(fnd)

        # not found in vectors for lang
        nfd = [(lang, word) for lang, word in seq
               if word not in vector_words[lang]]
        missing_vocab.add_sequence(nfd)

        if lang == 'en':
            continue

        # not found in vectors for lang but found in english
        fie = [(lang, word) for lang, word in nfd
               if word in vector_words['en']]
        english_vocab.add_sequence(fie)

# english fallback
"""
print(f'Current v_size={len(vocab)}')
print('Adding english fallback tokens')
for pair in english_vocab:
    if english_vocab.lookup(pair) >= 5:
        _, token = pair
        p = ('en', token)
        print(f'Fallback: {token}')
        vocab.add_token(p)
"""

print(f'Final v_size={len(vocab)}')
# sort the vocabularies
print('Sorting the vocabularies')
vocab.sort()
missing_vocab.sort()
english_vocab.sort()


vocab.save(VOCAB_SAVE)
missing_vocab.save(VOCAB_MISSING)
english_vocab.save(VOCAB_ENGLISH)
print(len(vocab))
