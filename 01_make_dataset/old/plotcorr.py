import matplotlib as mpl
mpl.use('Agg')
from dython.nominal import cramers_v, theils_u
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
style.use('fivethirtyeight')

"""
seee
https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
"""

DATAF = '/data/euprojects/silknow/tasks/dataset.place.csv'
ALL = ['place', 'dataset']
data_types = {k: 'str' for k in ALL}
usecols = ['index'] + ALL
df = pd.read_csv(DATAF, sep='\t', index_col='index',
                 usecols=usecols, dtype=data_types)
cv_place = cramers_v(df['dataset'], df['place'])
print(cv_place)
tu_place1 = theils_u(df['dataset'], df['place'])
tu_place2 = theils_u(df['place'], df['dataset'])
print(tu_place1)
print(tu_place2)

print('')

DATAF = '/data/euprojects/silknow/tasks/dataset.timespan.csv'
ALL = ['timespan', 'dataset']
data_types = {k: 'str' for k in ALL}
usecols = ['index'] + ALL
df = pd.read_csv(DATAF, sep='\t', index_col='index',
                 usecols=usecols, dtype=data_types)
cv_time = cramers_v(df['dataset'], df['timespan'])
print(cv_time)
tu_time1 = theils_u(df['dataset'], df['timespan'])
tu_time2 = theils_u(df['timespan'], df['dataset'])
print(tu_time1)
print(tu_time2)

"""
sns_plot = sns.heatmap(cv)
fig = sns_plot.get_figure()
fig.savefig("output.png")
"""
