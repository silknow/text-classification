import pandas as pd

ORIGINAL = '/data/euprojects/silknow/dataset.extracted.csv'
DST = '/data/euprojects/silknow/text.csv'
TASKS_SINGLE = ['timespan', 'place', 'material', 'technique']
MIN_COUNT = 50
# TASKS_MULTI = ['material']

df = pd.read_csv(ORIGINAL, sep='\t', usecols=['txt', 'lang'])
df.to_csv(DST, na_rep='null', sep='\t', index=False)
