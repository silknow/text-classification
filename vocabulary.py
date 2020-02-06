# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

"""
A vocabulary (aka dictionary) class.
"""

import logging
import csv
from collections import Counter

import numpy as np

from vechelper import load_vector_words
from textpreprocessor import sn_preprocess_text


class Vocab(object):
    def __init__(self, name='', pad='<PAD>', sos='<s>', eos='</s>',
                 unk='<unk>', specials=None):
        self.name = name
        self.pad = pad
        self.sos = sos
        self.eos = eos
        self.unk = unk
        self.specials = specials

        # ensure that pad exists
        if not self.pad:
            self.pad = '<PAD>'
            logging.warn("PAD token not specified - defaulting to `<PAD>`.")

        self.init_dicts()

    def init_dicts(self):
        self.token2id = {}
        self.id2token = {}
        self.token2count = {}

        # add reserved/special symbols but with 0 counts
        self.add_token(self.pad)
        self.token2count[self.pad] = 0
        if self.sos:
            self.add_token(self.sos)
            self.token2count[self.sos] = 0
        if self.eos:
            self.add_token(self.eos)
            self.token2count[self.eos] = 0
        if self.unk:
            self.add_token(self.unk)
            self.token2count[self.unk] = 0
        if self.specials:
            for special_token in self.specials:
                self.add_token(special_token)
                self.token2count[special_token] = 0

        self.reserved = len(self.token2id)
        self.reserved_tokens = [t for t in self.token2id]

    def __len__(self):
        """Returns the number of tokens in the dictionary."""
        return len(self.token2id)

    def __getitem__(self, i):
        """Returns the token of the specified id."""
        if not isinstance(i, int):
            raise ValueError('Invalid index type: {}'.format(type(i)))

        if i >= len(self.id2token) or i < 0:
            raise IndexError('The index (%d) is out of range.' % i)

        return self.id2token[i]

    def __contains__(self, w):
        """Returns whether a token is in the vocabulary."""
        return w in self.token2id

    def __iter__(self):
        return self.token2id.__iter__()

    def is_reserved(self, token):
        if token in self.reserved_tokens:
            return True
        return False

    def lookup(self, token, no_unk=False):
        """Lookup the id of a `token` if it is in the vocabulary.
        Otherwise return the index of the unknown token if it exits.
        Or None if it does not exist or the `no_unk` = True.
        """
        # return the index of the token if it is the vocabulary
        if token in self.token2id:
            return self.token2id[token]

        # else return the unknown token index
        if not no_unk and self.unk:
            return self.token2id[self.unk]

        # or None if no_unk=True or no unknown token exists
        return None

    def lookup_id(self, i, replace_error='<ERR:{}>'):
        """Lookup the token associated with the given id (same as vocab[i])
        except if the token does not exist, replace it with an error token.
        This function is useful for analysing sequences i.e. debugging.
        """
        token = None

        try:
            token = self.__getitem__(i)
        except (ValueError, IndexError) as e:
            if replace_error:
                token = replace_error.format(str(i))

        return token

    def add_token(self, token):
        """Add `token` to the vocabulary or increase its count."""
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
            self.token2count[token] = 1
        else:
            self.token2count[token] += 1

    def lookup_count(self, token):
        """Return token occurrence count.
        """
        return self.token2count.get(token)

    def add_sentence(self, sentence):
        """Add tokens in a sentence (space delimited list of tokens).
        Uses SOS and EOS if present in the vocabulary.
        """
        if not sentence:
            return
        if self.sos:
            self.add_token(self.sos)

        for token in sentence.split():
            self.add_token(token)

        if self.eos:
            self.add_token(self.eos)

    def add_sequence(self, sequence):
        """Add tokens in a sequence (e.g. list of tokens).
        Uses SOS and EOS if present in the vocabulary.
        """
        if not sequence:
            return
        if self.sos:
            self.add_token(self.sos)

        for token in sequence:
            self.add_token(token)

        if self.eos:
            self.add_token(self.eos)

    def __get_counts_for_reduce(self):
        """Returns a Counter.most_common() list of tuples without the special
        tokens. Used for functions that reduce the vocabulary.
        """
        counts = Counter(self.token2count)
        # protect special tokens by removing them from counter object
        for ii in range(self.reserved):
            token = self.lookup_id(ii)
            del counts[token]
        count_tuples = counts.most_common()
        return count_tuples

    def sort(self):
        """Sort the vocabulary.
        """
        self.reduce(min_freq=0, topk=0)

    def reduce(self, min_freq=0, topk=0):
        """Reduce the vocabulary size to at most `topk` > 0 entries by count
        and to tokens with count >= `min_freq`.
        If there are less than `topk` tokens with >= `min_freq`, the
        vocabulary will have less than `topk` tokens.

        Will sort the vocaulary.

        Returns total count of removed tokens.
        """

        count_tuples = self.__get_counts_for_reduce()
        unk_count = 0
        if self.unk:
            unk_count = self.token2count[self.unk]

        # reset the internal dictionaries (i.e. empty vocabulary)
        self.init_dicts()

        # eliminate of tokens that are not in the topk
        if topk > 0:
            # removed items will add to UNK count
            if self.unk:
                unk_count = sum([c for (_, c) in count_tuples[topk:]])

            # these are the topk entries:
            count_tuples = count_tuples[:topk]

        idx = len(self.token2id)
        # should now be equal to self.reserved
        assert idx == self.reserved

        for t, c in count_tuples:
            # if it's above or at the `min_freq` threshold
            if c >= min_freq:
                # add to the vocabulary
                self.token2id[t] = idx
                self.id2token[idx] = t
                self.token2count[t] = c
                idx = idx + 1
            # if it's below the threshold
            elif self.unk:
                # add to the UNK token count
                unk_count += c

        # add to the UNK count
        if self.unk:
            self.token2count[self.unk] += unk_count

        return unk_count

    def reduce_stopwords(self, stopwords):
        """Given a list of stopwords remove them from the vocabulary.
        Returns total count of removed tokens.
        """
        if not stopwords:
            return  # nothing to do

        # filter out stopwords not in the vocabulary
        stopwords = [tk for tk in stopwords if tk in self]
        if not stopwords:
            return  # all filtered out so there is nothing left to do here

        # get sum of stopwords occurrances in the training data
        stop_count = sum([self.token2count[tk] for tk in stopwords])

        count_tuples = self.__get_counts_for_reduce()

        # get current unk count
        unk_count = 0
        if self.unk:
            unk_count = self.token2count[self.unk]

        # reset the internal dictionaries (i.e. empty vocabulary)
        self.init_dicts()

        # set new unk count
        if self.unk:
            self.token2count[self.unk] += unk_count + stop_count

        # recreate the vocaublary w/o the stopwords
        idx = len(self.token2id)  # should now be equal to self.reserved
        for t, c in count_tuples:
            if t in stopwords:
                continue

            # add to the vocabulary
            self.token2id[t] = idx
            self.id2token[idx] = t
            self.token2count[t] = c
            idx = idx + 1

        return stop_count

    def reduce_intersect(self, vocab_b):
        """Reduces a vocabulary by keeping only the tokens that are also
        present in another vocabulary.
        Returns total count of removed tokens.
        """
        # find out which tokens are not present in vocab_b
        missing = [t for t in self.token2id
                   if (t not in vocab_b)
                   and (t not in self.reserved_tokens)]

        # remove them from this vocab
        c = self.reduce_stopwords(missing)
        return c

    def reduce_subtract(self, vocab_b):
        """Reduces a vocabulary by keeping only the tokens that are not
        present in another vocabulary.
        """
        # find out the common tokens to both vocabs
        present = [t for t in self.token2id
                   if (t in vocab_b)
                   and (t not in self.reserved_tokens)]

        # remove them from this vocab
        c = self.reduce_stopwords(present)
        return c

    def save(self, filepath):
        """Save vocabulary to a text file: token[space]count.
        The id will correspond to the line number.
        """
        with open(filepath, 'w') as fout:
            for ii in range(len(self.token2id)):
                token = self.id2token[ii]
                c = self.token2count[token]
                print('{} {}'.format(token, c), file=fout)

    def load(self, filepath):
        """Load vocabulary from a text file: token[space]count.
        The id will correspond to the line number.
        """
        with open(filepath) as fin:
            self.init_dicts()
            index = 0

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                token, c = line.split()
                c = int(c)

                self.token2id[token] = index
                self.id2token[index] = token
                self.token2count[token] = c

                index += 1
            # end of file
        return

    def sequence2ids(self, sequence, add_special=True):
        """Given a sequence, return the list with the indices of each token in
        it as a list.
        """
        ids = [self.lookup(token) for token in sequence]
        ids = [idx for idx in ids if idx is not None]

        if add_special:
            if self.sos:
                ids = [self.sos] + ids
            if self.eos:
                ids = ids + [self.eos]

        return ids

    def sentence2ids(self, sentence, add_special=True):
        """Given a sentence, return the list with the indices of each token in
        it as a list.
        """
        ids = self.sequence2ids(sentence.split(), add_special)

        return ids

    def ids2sentence(self, ids, remove_special=False):
        if remove_special:
            return ' '.join(self.lookup_id(i) for i in ids
                            if i >= self.reserved)

        return ' '.join(self.lookup_id(i) for i in ids)

    def ids2sequence(self, ids):
        return [self.lookup_id(i) for i in ids]

    def load_from_embeddings(self, filepath, skip_header=True):
        """Load vocab from an embeddings file - no counts present."""
        self.init_dicts()

        with open(filepath) as fin:
            if skip_header:
                next(fin)

            for line in fin:
                values = line.strip().split()
                token = values[0]
                self.add_token(token)
                self.token2count[token] = 0

        return

    def calc_reduce_str(self, min_freq=5, topk=50000, p=3):
        """Calculate vocabulary size and coverage given defined reductions:
            percentile (p)
            min_count
            topk tokens
        Returns a string to be displayed.
        """
        reduce_strings = []
        # current total (no reduction)
        counts_total = list(self.token2count.values())
        vsize_total = len(self)
        tokens_total = sum(counts_total)
        min_count_total = min(counts_total)
        s = ('Current\n'
             f'\tvsize={vsize_total}\n'
             f'\ttokens={tokens_total}\n'
             f'\tmin_count={min_count_total}\n')
        reduce_strings.append(s)

        # min_count (if tokens below min count where removed)
        counts_min = [c for c in counts_total if c >= min_freq]
        tokens_min = sum(counts_min)
        coverage_min = (tokens_min / tokens_total) * 100
        vsize_min = len(counts_min)
        s = ('min\n'
             f'\tvsize={vsize_min}\n'
             f'\ttokens={tokens_min}\n'
             f'\tmin_count={min_freq}\n'
             f'\tcoverage={coverage_min:.1f}\n')
        reduce_strings.append(s)

        # topk (if only the topk tokens where preserved)
        counts_topk = sorted(counts_total, reverse=True)[:topk]
        tokens_topk = sum(counts_topk)
        coverage_topk = tokens_topk / tokens_total
        vsize_topk = topk
        min_count_topk = min(counts_topk)
        s = ('topk\n'
             f'\tvsize={vsize_topk}\n'
             f'\ttokens={tokens_topk}\n'
             f'\tmin_count={min_count_topk}\n'
             f'\tcoverage={coverage_topk:.1f}\n')
        reduce_strings.append(s)

        # percentile (if tokens below percentile where removed)
        min_count_percentile = np.percentile(counts_total, p)
        counts_percentile = [c for c in counts_total
                             if c >= min_count_percentile]
        tokens_percentile = sum(counts_percentile)
        coverage_percentile = (tokens_percentile / tokens_total) * 100
        vsize_percentile = len(counts_percentile)
        s = ('Percentile\n'
             f'\tvsize={vsize_percentile}\n'
             f'\ttokens={tokens_percentile}\n'
             f'\tmin_count={min_count_percentile}\n'
             f'\tcoverage={coverage_percentile:.1f}\n')
        reduce_strings.append(s)

        # done
        return '\n'.join(reduce_strings)


class VocabMultiLingual(Vocab):
    """Entries in this vocabulary are (lang, token) pairs
    """

    def __init__(self, name='', pad=('-', '<PAD>'), sos=('-', '<s>'),
                 eos=('-', '</s>'), unk=('-', '<unk>'), specials=None):
        super().__init__(name, pad, sos, eos, unk, specials)

    def save(self, filepath):
        """Save vocabulary to a text file: lang[space]token[space]count.
        The id will correspond to the line number.
        """
        with open(filepath, 'w') as fout:
            for ii in range(len(self.token2id)):
                pair = self.id2token[ii]
                lang, token = pair

                c = self.token2count[pair]
                print('{} {} {}'.format(lang, token, c), file=fout)

    def load(self, filepath):
        """Load vocabulary from a text file: lang[space]token[space]count.
        The id will correspond to the line number.
        """
        with open(filepath) as fin:
            self.init_dicts()
            index = 0

            for line in fin:
                line = line.strip()
                if not line:
                    raise ValueError('Empy line in vocab file')
                lang, token, c = line.split()
                pair = (lang, token)
                c = int(c)
                self.token2id[pair] = index
                self.id2token[index] = pair
                self.token2count[token] = c

                index += 1
            # end of file
        return

    def load_from_embeddings(self, lang_file_pairs, skip_header=True):
        """Load vocab from an embeddings files - no counts present.
        lang_file_pairs = (lang, filepath)
        """
        self.init_dicts()

        for lang, filepath in lang_file_pairs:
            with open(filepath) as fin:
                if skip_header:
                    next(fin)

                for line in fin:
                    values = line.strip().split()
                    token = (lang, values[0])
                    self.add_token(token)
                    self.token2count[token] = 0

            return

    def add_from_monolingual(self, vocab_mono, lang):
        for token in vocab_mono:
            if vocab_mono.is_reserved(token):
                continue
            pair = (lang, token)
            self.add(pair)

    @staticmethod
    def sentence_to_multilingual_sequence(sentence, lang):
        tokens = sentence.split()
        return [(lang, token) for token in tokens]

    @staticmethod
    def sentence_to_monolingual(sentence_pairs):
        tokens = [token for _, token in sentence_pairs]
        return ' '.join(tokens)


def sn_create_vocab(embeddings_files, data_file=None):
    """SilkNOW create vocabulary.

    embeddings_files: a dictionary of lang -> embeddings file.
    data_file (optional, str): a TSV file containing text and lang fields.

    Returns: a VocabMultiLingual object.
    """
    vocab = VocabMultiLingual(sos=None, eos=None, unk=None)

    # load vector words
    vector_words = {}
    if embeddings_files is None:
        vector_words = None
    else:
        for lang in embeddings_files:
            vector_words[lang] = set(load_vector_words(embeddings_files[lang]))

    # create vocabulary only from embeddings
    if data_file is None:
        for lang in embeddings_files:
            words = list(vector_words[lang])
            for word in words:
                vocab.add((lang, word))
        return vocab

    # create vocabulary from text
    with open(data_file) as finp:
        reader = csv.DictReader(finp, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            text = row['txt']
            lang = row['lang']

            text = sn_preprocess_text(text, lang)
            seq = vocab.sentence_to_multilingual_sequence(text, lang)

            # filter with vectors
            seq = [(lang, word) for lang, word in seq
                   if word in vector_words[lang]]
            # add
            vocab.add_sequence(seq)

    return vocab
