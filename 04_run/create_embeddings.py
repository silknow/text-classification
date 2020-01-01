#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

import torch
import numpy as np
from vocabulary import VocabMultiLingual
from vechelper import load_vectors, save_vectors_multilingual

# DST = '../model/embedding_matrix.pt'
# VOCAB = '../dataset/vocab.txt'

DST = '/data/euprojects/silknow/embedding_matrix.pt'
VOCAB = '/data/euprojects/silknow/vocab.txt'

VECTORS = {
    'en': '/data/vectors/ftaligned/wiki.en.align.vec',
    'ca': '/data/vectors/ftaligned/wiki.ca.align.vec',
    'es': '/data/vectors/ftaligned/wiki.es.align.vec',
    'fr': '/data/vectors/ftaligned/wiki.fr.align.vec',
}
vector_size = 300

vocab = VocabMultiLingual(sos=None, eos=None, unk=None)
vocab.load(VOCAB)
vocab_size = len(vocab)
print(f'v_size: {vocab_size}')

found = set()

shape = (vocab_size, vector_size)
embedding_matrix = np.zeros(shape, dtype=np.float32)
loaded = 0

debug_dict = {}


for lang in VECTORS:
    lang_loaded = 0
    print(lang)
    vector_filepath = VECTORS[lang]
    lang_vectors = load_vectors(vector_filepath)

    for token in lang_vectors:
        pair = (lang, token)
        if pair not in vocab:
            continue
        index = vocab.lookup(pair, no_unk=True)
        if not index:
            continue
        found.add(pair)
        debug_dict[pair] = lang_vectors[token]
        embedding_matrix[index] = lang_vectors[token]
        lang_loaded += 1
    print(f'Loaded for {lang}=\t{lang_loaded}')

    loaded += lang_loaded
print(f'Total Loaded:\t{loaded}')

tensor = torch.from_numpy(embedding_matrix)
torch.save(tensor, DST)

save_vectors_multilingual('/data/euprojects/silknow/debug.vec',
                          debug_dict, 300)

for p in vocab:
    if p not in found:
        print(p)
