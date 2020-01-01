#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >
"""
Text preprocessing script
"""
import csv
from functools import partial
from sacremoses import MosesTokenizer

from textpreprocessor import compose_f
from textpreprocessor import normalize_nfkc
from textpreprocessor import normalize_whitespace, clean_html, replace_string
from textpreprocessor import (strip_non_printing, normalize_quotes_and_dashes,
                              replace_punct)
from textpreprocessor import strip_symbols, case_fold

"""
SRC_FILE = "/home/rei/Dropbox/Work/EUProjects/silknow/dataset/dataset.csv"
DST_FILE = "/home/rei/Dropbox/Work/EUProjects/silknow/dataset/dataset.prp.csv"
TXT_FIELD = 'text'
"""

SRC_FILE = '/data/euprojects/silknow/dataset.mapped.csv'
DST_FILE = '/data/euprojects/silknow/dataset.prp.csv'
TXT_FIELD = 'txt'


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


def create_preprocessing_function():
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


def main():
    preprocess = create_preprocessing_function()

    # dataset
    print('dataset')
    with open(SRC_FILE) as src_file, open(DST_FILE, 'w') as dst_file:
        # open source csv and read header
        reader = csv.DictReader(src_file, delimiter='\t',
                                quoting=csv.QUOTE_NONE)
        row = next(reader)
        fieldnames = list(row.keys())
        # open destination csv and write header
        writer = csv.DictWriter(dst_file, delimiter='\t',
                                quoting=csv.QUOTE_NONE, fieldnames=fieldnames)
        writer.writeheader()

        # preprocess - write - read loop
        while True:
            if not row:
                break
            text = row[TXT_FIELD]
            lang = row['lang']

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

            # write
            row[TXT_FIELD] = text
            writer.writerow(row)

            # read
            try:
                row = next(reader)
            except StopIteration:
                break


if __name__ == '__main__':
    main()
