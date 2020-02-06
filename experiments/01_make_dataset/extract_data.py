#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >
from collections import defaultdict
from datahelper import read_i_json, write_csv, lang_filter, min_token_filter
from datahelper import dedup_records, add_id
from datamaps import read_label_map, apply_label_map, save_list_txt
from datamaps import premap_time


MIN_CHARS = 50
MAX_LABEL_CHARS = 70
MIN_TOKENS = 15
dst_e = '/data/euprojects/silknow/dataset.extracted.csv'
dst_m = '/data/euprojects/silknow/dataset.mapped.csv'
dst_u = '/data/euprojects/silknow/temp'

VAM_CONFIG = {
    'place': 'place',
    'timespan': 'date_text',
    'material': 'materials',
    'technique': 'techniques',
    '_txt': ['descriptive_line', 'physical_description',
             'public_access_description', 'historical_context_note',
             'history_note'],
    '_lang': 'en',
    '_dataset': 'vam',
    '_data': '/data/euprojects/silknow/crawl_new/vam/records',
    '_chars': MIN_CHARS,
    '_lbl_chars': MAX_LABEL_CHARS,
    '_ext': '*.json',
}

IMATEX_CONFIG = {
    'place': 'ORIGEN*',
    'material': 'MATÈRIES*',
    'technique': "TÈCNICA*",
    'timespan': 'CRONOLOGIA*',
    '_txt': ['DESCRIPCIÓ', 'DESCRIPCIÓ TÈCNICA'],
    '_lang': 'ca',
    '_data': '/data/euprojects/silknow/crawl_new/imatex/records',
    '_chars': MIN_CHARS,
    '_lbl_chars': MAX_LABEL_CHARS,
    '_ext': '*_ca.json',
    '_dataset': 'imatex'
}

"""
IMATEX_CONFIG = {
    'place': 'ORIGEN*',
    'material': 'MATÈRIES*',
    'technique': "TÈCNICA*",
    'timespan': 'CRONOLOGIA*',
    '_txt': ['DESCRIPCIÓN', 'DESCRIPCIÓN TÈCNICA'],
    '_lang': 'es',
    '_data': '/data/euprojects/silknow/crawl_new/imatex/records',
    '_chars': MIN_CHARS,
    '_lbl_chars': MAX_LABEL_CHARS,
    '_ext': '*_es.json',
    '_dataset': 'imatex'
}
"""

JOCONDE_CONFIG = {
    'material': 'Matériaux/techniques',
    'technique': 'Matériaux/techniques',
    'timespan': 'Période création/exécution',
    '_txt': ['Description', 'Genèse', 'Historique',
             'Précision sujet représenté', 'Titre'],
    '_lang': 'fr',
    '_dataset': 'joconde',
    '_data': '/data/euprojects/silknow/crawl_new/joconde/records',
    '_chars': MIN_CHARS,
    '_lbl_chars': MAX_LABEL_CHARS,
    '_ext': '*.json',
}
# 'subject': 'Sujet représenté',
# 'technique': 'Matériaux/techniques',
# Always France:
# 'place': 'Lieu création / utilisation',

CERES_CONFIG = {
    'material': 'Materia/Soporte',
    'technique': 'Técnica',
    'place': 'Lugar de Producción/Ceca',
    'timespan': 'Datación',
    '_txt': ['Descripción', 'Clasificación Razonada'],
    '_lang': 'es',
    '_dataset': 'ceres',
    '_data': '/data/euprojects/silknow/crawl_new/ceres-mcu/records',
    '_chars': MIN_CHARS,
    '_lbl_chars': MAX_LABEL_CHARS,
    '_ext': '*.json',
}


MET_CONFIG = {
    'material': 'Medium:',
    'technique': 'Classification:',
    'place': 'Culture:',
    'timespan': 'date',
    '_txt': ['description'],
    '_lang': 'en',
    '_dataset': 'met',
    '_data': '/data/euprojects/silknow/crawl_new/met-museum/records',
    '_chars': MIN_CHARS,
    '_lbl_chars': MAX_LABEL_CHARS,
    '_ext': '*.json',
}

MFA_CONFIG = {
    'material': 'mediumOrTechnique',
    'technique': 'mediumOrTechnique',
    'place': 'teaser',
    'timespan': 'teaser',
    '_txt': ['description'],
    '_lang': 'en',
    '_dataset': 'mfa',
    '_data': '/data/euprojects/silknow/crawl_new/mfa-boston/records',
    '_chars': MIN_CHARS,
    '_lbl_chars': MAX_LABEL_CHARS,
    '_ext': '*.json',
}


CONFIGS = [VAM_CONFIG, IMATEX_CONFIG, JOCONDE_CONFIG, CERES_CONFIG,
           MET_CONFIG, MFA_CONFIG]


def count_field(records, field):
    d = defaultdict(int)
    for r in records:
        if r.get(field):
            d[r['dataset']] += 1
    return d


if __name__ == '__main__':
    records = []
    for config in CONFIGS:
        recs = read_i_json(config)
        n1 = len(recs)
        if config['_dataset'] == 'imatex':
            recs = lang_filter(recs)
        n2 = len(recs)
        s = '{}: {} (unfiltered: {})'
        s = s.format(config['_dataset'], n2, n1)
        print(s)
        records += recs
    print('total: {}'.format(len(records)))

    # deduplication
    records = dedup_records(records)
    print('dedup total: {}'.format(len(records)))
    write_csv(dst_e, records)

    print('\n---------------------\n')
    # place
    print('place')
    print(count_field(records, 'place'))
    place_map = read_label_map('maps/place.txt')
    map_y, map_n = apply_label_map(records, 'place', place_map, delete=True)
    c = len(map_y)
    save_list_txt(map_n, dst_u, 'place.txt')
    print('place modified: {}'.format(c))

    # time
    print('timespan')
    print(count_field(records, 'timespan'))
    p_c = premap_time(records)
    time_map = read_label_map('maps/timespan.txt')
    map_y, map_n = apply_label_map(records, 'timespan', time_map, delete=True)
    c = len(map_y)
    save_list_txt(map_n, dst_u, 'time.txt')
    print(f'timespan modified: {c} (premap: {p_c})')

    # material
    print('material')
    print(count_field(records, 'timespan'))
    material_map = read_label_map('maps/material.txt')
    map_y, map_n = apply_label_map(
        records, 'material', material_map, delete=True)
    c = len(map_y)
    save_list_txt(map_n, dst_u, 'material.txt')
    print('material modified: {}'.format(c))

    # technique
    technique_map = read_label_map('maps/technique.txt')
    map_y, map_n = apply_label_map(records, 'technique',
                                   technique_map, delete=True)
    c = len(map_y)
    save_list_txt(map_n, dst_u, 'technique.txt')
    print('technique modified: {}'.format(c))

    print('\n---------------------\n')
    # MIN TOKENS
    records = min_token_filter(records, MIN_TOKENS)
    print(len(records))

    # add id
    add_id(records)

    # write
    write_csv(dst_m, records)
