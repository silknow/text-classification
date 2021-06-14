import os
from collections import Counter
import datetime


def read_label_map(description2label_txt):
    """Reads a label map text file into a dictionary (map)
    """
    description2label = {}
    f = open(description2label_txt, 'r')
    s = f.read()
    lines = s.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue

        # convert line to fields
        fields = line.split(';')
        if len(fields) < 2:
            continue

        # read description
        description = fields[0].strip().lower()
        description = description.replace('#@#', ';')

        # read label
        label = fields[1].strip()
        if label.lower() == 'nan':
            label = 'null'
        label = label.replace(' ', '_').replace('/', ',')

        # add to dict
        description2label[description] = label.strip().lower()

    return description2label


def apply_label_map(records, field, label_map, delete=False):
    mapped = []
    not_mapped = []
    for record in records:
        # exists and has value?
        if not record.get(field):
            continue

        # matched?
        if record[field] in label_map:
            # yes
            mapped.append(record[field])
            record[field] = label_map[record[field]]

        else:
            # no
            not_mapped.append(record[field])
            if delete:
                record[field] = 'null'

    # return
    not_mapped = Counter(not_mapped).most_common()
    return mapped, not_mapped


def save_list_txt(save_list, path, name):
    filepath = os.path.join(path, name)
    with open(filepath, 'w') as fout:
        for v, c in save_list:
            print(f'{c}#{v}', file=fout)


def extract_date_year(txt):
    try:
        date = datetime.datetime.strptime(txt, "%Y")
        century = str(date.year - 1)[:2]
        return century
    except ValueError:
        pass
    return None


def extract_date_year_span(txt):
    txt = txt.replace('=', '-')
    sp = txt.split('-')
    if sp != 2:
        return None

    date_start = None
    try:
        date = datetime.datetime.strptime(sp[0].strip(), "%Y")
        date_start = str(date.year - 1)
    except ValueError:
        return None

    date_end = None
    try:
        date = datetime.datetime.strptime(sp[1].strip(), "%Y")
        date_end = str(date.year - 1)
    except ValueError:
        return None

    # if same century
    if date_start[0:2] == date_end[0:2]:
        return date_start[0:2]

    return None


def premap_time(records, field='timespan'):
    c = 0
    for record in records:
        txt = record.get(field)
        # exists and has value?
        if not txt:
            continue
        txt = txt.replace('[ca]', '')
        txt = txt.replace('ca.', '')
        txt = txt.replace('(made)', '')
        txt = txt.replace('(made)', '')
        txt = txt.replace('proably frenchabout', '')
        txt = txt.replace('frenchabout', '')
        txt = txt.replace('probably englishabout', '')
        txt = txt.replace('englishabout', '')
        txt = txt.replace('english', '')
        txt = txt.replace('french', '')
        txt = txt.replace('spanish', '')
        txt = txt.replace('swedish', '')
        txt = txt.replace('dutch', '')
        txt = txt.strip()

        date = extract_date_year(txt)
        if date:
            record[field] = date
            c += 1
            continue

        date = extract_date_year_span(txt)
        if date:
            record[field] = date
            c += 1
            continue

        c += 1
    return c
