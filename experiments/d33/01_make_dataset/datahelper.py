import os
import glob
import json
import langid
import pandas as pd


def __process_value(value, truncate, lower):
    if type(value) == list:
        value = ' '.join(value)
    if lower:
        value = value.strip().lower()
    value = ' '.join(value.split())
    if truncate > 0:
        value = value[0:truncate].strip()
    return value


def read_json_value(json_data, key, truncate=50, lower=True):
    """Read a value for a certain key in a silknow json record."""
    value = None

    # find the json field that contain the key inside the 'fields' array
    for field in json_data['fields']:
        if field["label"] == key:
            if 'value' in field:
                value = field['value']
            elif 'values' in field:
                value = field['values']
            else:
                continue

            if not value:
                return None

            value = __process_value(value, truncate, lower)
            return value

    # find the json field that contain the key outside the 'fields' array
    for k in json_data.keys():
        if k == key:
            value = json_data[key]
            if not value:
                return None
            value = __process_value(value, truncate, lower)
            return value

    return value


def read_json_text(json_data, txt_keys):
    """Extracts text data from file, concatenates all text fields."""
    # read from file
    txt_data = []
    for txt_key in txt_keys:
        val = read_json_value(json_data, txt_key, truncate=0, lower=False)
        if val:
            txt_data.append(val)

    txt = ' '.join(txt_data)
    txt = txt.strip()

    # normalize text whitespace
    txt = ' '.join(txt.split())

    return txt


def read_json_labels(json_data, label_map, truncate):
    """Extracts categorical data from JSON crawl files,"""
    record = {}

    for class_name, key in label_map:
        record[class_name] = read_json_value(json_data, key, truncate)

    return record


def read_json_file(fpath, json_config):
    """Reads a JSON crawl file."""
    with open(fpath, 'rb') as reader:
        json_data = json.loads(reader.read(), encoding='utf-8')

        # txt
        txt = read_json_text(json_data, json_config['_txt'])
        if not txt:
            return None
        if len(txt) < json_config['_chars']:
            return None

        # labels
        label_map = [x for x in json_config.items()
                     if not x[0].startswith('_')]
        record = read_json_labels(json_data, label_map,
                                  json_config['_lbl_chars'])

        # add txt to record
        record['txt'] = txt

        # add other properties
        record['dataset'] = json_config['_dataset']
        record['filename'] = os.path.split(fpath)[-1]
        record['lang'] = json_config['_lang']

        return record
    return None


def read_i_json(json_config):
    """Reads files into a list of dictionaries
    """
    data = []

    data_dir = json_config['_data']
    extension = json_config['_ext']

    # for each file in the data dir with the data sufix
    for i_json in glob.glob(os.path.join(data_dir, extension)):
        record = read_json_file(i_json, json_config)
        if record:
            data.append(record)

    return data


def dedup_records(records):
    df = pd.DataFrame(records).drop_duplicates()
    return df.to_dict('records')


def lang_filter(records):
    return [r for r in records if langid.classify(r['txt'])[0] == r['lang']]


def min_token_filter(records, min_tokens):
    return [r for r in records if len(r['txt'].split()) >= min_tokens]


def add_id(records):
    idx = 1
    for r in records:
        r['ecode'] = 'e#' + str(idx)
        idx += 1


def write_csv(dst, records):
    df = pd.DataFrame(records)
    df.to_csv(dst, na_rep='null', sep='\t', index=False)
