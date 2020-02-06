#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

"""
Embeddings helper functions
"""
import io
import unicodedata
import numpy as np


def load_vectors(fname):
    """Load embeddings."""
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        token = tokens[0].strip()
        token = unicodedata.normalize('NFKC', token)
        data[token] = list(map(float, tokens[1:]))
    return data


def load_vector_words(fname):
    """Load only the words from the embeddings file."""
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    words = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        token = tokens[0].strip()
        token = unicodedata.normalize('NFKC', token)
        words.append(token)
    return words


def save_vectors(fpath, data, vector_size):
    """
    Export embeddings (in token -> [float] format to a text file
    """
    vocab_size = len(data)
    with io.open(fpath, 'w', encoding='utf-8') as f:
        f.write(u"%i %i\n" % (vocab_size, vector_size))
        for token in data:
            vec = " ".join('%.5f' % x for x in data[token])
            f.write(u"%s %s\n" % (token, vec))


def save_vectors_multilingual(fpath, data, vector_size):
    """
    Export embeddings (in lang[space]token[space][vector] format to a text file
    """
    vocab_size = len(data)
    with io.open(fpath, 'w', encoding='utf-8') as f:
        f.write(u"%i %i\n" % (vocab_size, vector_size))
        for pair in data:
            vec = " ".join('%.5f' % x for x in data[pair])
            lang, token = pair
            f.write(u"%s %s %s\n" % (lang, token, vec))


def sn_create_embeddings(embeddings_files, vocab, vector_size=300):
    """SilkNOW create embeddings.

    embeddings_files: a dictionary of lang -> embeddings file.
    vocab: the VocabMultiLingual

    Returns: a torch tensor.
    """
    found = set()

    vocab_size = len(vocab)
    shape = (vocab_size, vector_size)
    embedding_matrix = np.zeros(shape, dtype=np.float32)

    for lang in embeddings_files:
        print(lang)
        vector_filepath = embeddings_files[lang]
        lang_vectors = load_vectors(vector_filepath)

        for token in lang_vectors:
            pair = (lang, token)
            if pair not in vocab:
                continue
            index = vocab.lookup(pair, no_unk=True)
            if not index:
                continue
            found.add(pair)

            embedding_matrix[index] = lang_vectors[token]

    import torch
    tensor = torch.from_numpy(embedding_matrix)
    return tensor
