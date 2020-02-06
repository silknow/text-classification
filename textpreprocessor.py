#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

"""
Text preprocessing functions.
"""
import unicodedata
import re
from functools import reduce, partial

import ftfy
from sacremoses import MosesTokenizer
from unidecode import unidecode

#
# CONSTANTS
#
NUMBER_WORDS_EN = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
}

NUMBER_WORDS_CA = {
    '1': 'un',
    '2': 'dos',
    '3': 'tres',
    '4': 'quatre',
    '5': 'cinc',
    '6': 'sis',
    '7': 'set',
    '8': 'vuit',
    '9': 'nou',
}

NUMBER_WORDS_ES = {
    '1': 'uno',
    '2': 'dos',
    '3': 'tres',
    '4': 'cuatro',
    '5': 'cinco',
    '6': 'seis',
    '7': 'siete',
    '8': 'ocho',
    '9': 'nueve',
}

RPLS = {
    'ssilk': 'silk',
    'inches': 'inches',
    'length': 'length',
    'jaqard': 'jacquard',
    'jaquard': 'jacquard',
    'italiantextile': 'italian textile',
    'needleweaving': 'needle weaving',
    'waistseam': 'waist seam',
    'sleeveband': 'sleeve band',
    'drawnwork': 'drawn work',
    'needlework': 'needle work',
    'needlewoven': 'needle woven',
    'threadwork': 'thread work',
    'needlecase': 'needle case',
    'longsleeve': 'long sleeve',
    'designerembroidery': 'designer embroidery',
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
    'lampàs': 'lampas',
    'esfumaturas': 'esfumado',
    'madrids': 'madrid',
}
RPLS_EN = {
    'botehs': 'boteh',
    'halbedier': 'halberdier',
    'manuscruipt': 'manuscript',
    'latchets': 'latchet',
    'lustring': 'calendering',
    'unplied': 'not plied',
    'cannellé': 'canelle',
    'canellé': 'canelle',
    'clothiing': 'clothing',
    'bizantinos': 'byzantine',
    'backseam': 'back seam',
    'unembroidered': 'not embroidered',
    'emboidered': 'embroidered',
    'floorspread': 'floor spread',
    'overknit': 'over knit',
    'overstitch': 'over stitch',
    'underbodice': 'under bodice',
    'undersleeve': 'under sleeve',
    'handscreens': 'hand screens',
    'backstitched': 'back stitched',
    'regiion': 'region',
    'lisere': 'edging',
    'laceing': 'lacing',
    'commmission': 'commission',
}
RPLS_ES = {
    'espolinadas': 'brocado',
    'espolinada': 'brocado',
    'espolinado': 'brocado',
    'brochadas': 'brocado',
    'brochada': 'brocado',
    'esfumaturas': 'esfumado',
    'esfumatura': 'esfumado',
    'lampàs': 'lampas',
    'éventails': 'eventail',
    'beentjes': 'beentje',
    'abanos': 'abano',
}
RPLS_CA = {
    'espolinadas': 'brocades',
    'espolinades': 'brocades',
    'espolinat': 'brocades',
    'espolinada': 'brocades',
    'brochadas': 'brocades',
    'lampàs': 'lampas',
    'intensidad': 'intensitat',

}


cleanr = re.compile('<.*?>')
cleanr_e = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


def compose_f(funcs):
    """Given a list of functions `funcs`=[f1, f2, f3, ..., fn] returns their
    composition:
        f = fn(...f3(f2(f1()))...)
    """
    funcs = reversed(funcs)
    # reduce applies a function of 2 arguments (the first argument of reduce)
    # cumulatively to the items of the iterable (second argument) from
    # left to right. The left argument (here `f`) is the accumulated value
    # and the right argument (here `g`) is the update value from the iterable.
    # Each `g` value from the list is a function. The applied function
    # (the first lambda) returns a function that is the composition of its two
    # arguments (`f(g(x))`. The initializer i.e. the first value of `f` is the
    # identity function (second lambda).
    return reduce(lambda f, g: lambda x: f(g(x)), funcs, lambda x: x)


def char_name(char):
    """Returns the name of a character or empty string"""
    try:
        return unicodedata.name(char).lower()
    except ValueError:
        return ''
    return ''


def char_is_punct(char):
    """True if char is a punctuation character."""
    return unicodedata.category(char).startswith('P')


def fix_text(text):
    """Fixes text issues using ftfy including mojibake and html entities.
    Returns NFC normalized unicode text."""
    return ftfy.fix_text(text)


def clean_html(s):
    """Uses a regular expression to remove leftover html tags in text.
    """
    s = re.sub(cleanr, '', s)
    s = re.sub(cleanr_e, '', s)
    return s


def normalize_quotes_and_dashes(text):
    """Normalizes many of the quotations and dashes.
    Note: Some of these are not handled by unidecode transliteration."""
    res = []
    # fix dashes
    for c in text:
        if not unicodedata.category(c) == 'Pd':  # "Punctuation, Dash"
            res.append(c)
        else:
            res.append('-')
    text = res

    # fix quotes
    # see https://en.wikipedia.org/wiki/Quotation_mark
    # this should cover most common cases
    res = []
    for c in text:
        if not unicodedata.category(c) in ['Pe', 'Ps', 'Pf', 'Pi']:
            res.append(c)
        else:
            name = char_name(c)
            if ('quot' in name or 'corner bracket' in name or
                    'double' in name or 'fullwidth' in name):
                res.append('"')  # replace
            else:
                res.append(c)
    text = ''.join(res)
    # dumb quotes
    text = re.sub(r"''", r'"', text)
    text = re.sub(r"``", r'"', text)
    text = re.sub(r"<<", r'"', text)
    text = re.sub(r">>", r'"', text)

    return text


def strip_accents(text):
    """Strips accents from text."""
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text


def ascii_fold(text):
    """Converts all characters in text to ascii (ascii transliteration)."""
    return unidecode(text)


def strip_punct(text):
    """Strips punctuation (including quotes and dash) from text.
    Beware:
        john's -> johns
        zero-tolerance -> zerotolerance
        example.com -> examplecom
    """
    return ''.join(c for c in text if not char_is_punct(c))


def replace_punct(text, replacement=' ', protect=''):
    """Replaces punctuation (including quotes and dash) from text.
    If using whitespace (default) it is probably a good idea to call
    normalize_whitespace after.
    e.g.:
        doctor, who? -> doctor  who
    """
    return ''.join(c if (not char_is_punct(c)) or (c in protect)
                   else replacement for c in text)


def replace_string(text, replacements=None, whitespace=True):
    """A wrapper around str.replace where replacements is a dictionary:
        original_string -> replacement_string
    whitespace=True surounds the replacement with whitespaces.
    """
    if not replacements:
        return text
    for ori, rpl in replacements.items():
        if whitespace:
            rpl = ' ' + rpl + ' '
        text = text.replace(ori, rpl)
    return text


def normalize_unicode_punctuation(text):
    """Converts (transliterates) unicode punctuation, symbols,
    and numbers to their ASCII equivalent.
    """
    res = []
    for c in text:
        cat = unicodedata.category(c)[0]
        if cat in ['P', 'N']:
            c = unidecode(c)
        elif cat == 'S':
            c = unidecode(c)
            if len(c) > 1:
                # most commonly these will be things like EUR, (tm) and (r)
                # we want to add a space before, if it is extra it will
                # be removed by whitespace normalization
                c = ' ' + c

        if c:
            res.append(c)
    return ''.join(res)


def strip_non_printing(text):
    """Removes non-printing (including control) characters from text.
    (same as moses script).
    """
    return ''.join([c for c in text if c.isprintable()])


def strip_symbols(text):
    """Removes symbols (unicode category 'S').
    """
    return ''.join(c for c in text
                   if unicodedata.category(c)[0] != 'S')


def strip_digits(text):
    """Removes digits (number) charaters.
    """
    return ''.join(c for c in text if not c.isdigit())


def case_fold(text):
    """Converts text to lower case."""
    return text.lower()


def normalize_whitespace(text):
    """Merges multiple consecutive whitespace characthers converting them to
    space (` `). Also strips whitespaces from start and end of the text."""
    return ' '.join(text.split())


def str_strip(text):
    """Strips whitespaces from the start and the end of the text."""
    return text.strip()


def str_lstrip(text):
    """Strips whitespaces from thestart of the text."""
    return text.lstrip()


def str_rstrip(text):
    """Strips whitespaces from the end of the text."""
    return text.rstrip()


def normalize_nfc(text):
    """Applies NFC normalization."""
    return unicodedata.normalize('NFC', text)


def normalize_nfd(text):
    """Applies NFD normalization."""
    return unicodedata.normalize('NFD', text)


def normalize_nfkc(text):
    """Applies NFKC normalization e.g. ™ to TM, ..."""
    return unicodedata.normalize('NFKC', text)


def normalize_nfkd(text):
    """Applies NFKD normalization."""
    return unicodedata.normalize('NFKD', text)


def sn_preprocessing_function():
    """Returns the SilkNOW text preprocessing function.

    Return: function to be used to preprocess text.
    """
    punct_f1 = partial(replace_punct, replacement='', protect="'-")
    punct_f2 = partial(replace_punct, replacement=' ', protect="'")
    punct_f3 = partial(replace_punct, replacement=" ' ", protect="")
    replace_f = partial(replace_string, replacements=RPLS)

    funcs = [
        case_fold,
        replace_f,
        strip_non_printing,
        clean_html,
        normalize_quotes_and_dashes,
        punct_f1,
        punct_f2,
        punct_f3,
        strip_symbols,
        normalize_nfkc,
        normalize_whitespace,
    ]

    return compose_f(funcs)


def sn_preprocess_text(text, lang='en'):
    """SilkNOW: preprocess and tokenize text.
    Returns preprocessed and tokenized text as a space delimited sequence of
    tokens (i.e. returns a string).
    """
    preprocess = sn_preprocessing_function()

    # tokenize
    tkz = MosesTokenizer(lang=lang)
    text = tkz.tokenize(text, return_str=True, escape=False)
    if lang == 'ca':
        text = replace_string(text, replacements=RPLS_CA)
    elif lang == 'es':
        text = replace_string(text, replacements=RPLS_ES)
    elif lang == 'en':
        text = replace_string(text, replacements=RPLS_EN)

    # preprocess
    text = preprocess(text)

    return text
