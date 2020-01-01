import os
from ihelper import read_i_json, read_label_map, convert_labels
from ihelper import remove_short_texts, remove_duplicates
from ihelper import ex_p_class_print, ex_p_class_min
from ihelper import split_save_data


key_txt = 'physical_description'
key_date = 'date_text'
# key_materials = 'materials_techniques'
key_site = 'place'

data_dir = '../data/vam/records'
dataset_dir = '../datasets/vam'
map_dir = '../maps/vam'
MIN_TOKENS = 5
MIN_EX = 50
TEST_EX = 0.4
SEED = 1984

site_label_map = read_label_map(os.path.join(map_dir, 'site_map.txt'))
data_site = read_i_json(data_dir, '*.json', key_txt, key_site)
data_site = remove_short_texts(data_site, MIN_TOKENS)
data_site, err_site = convert_labels(data_site, site_label_map)
data_site = remove_duplicates(data_site)
ex_p_class_print(data_site)
print('-' * 80)
data_site = ex_p_class_min(data_site, MIN_EX)
ex_p_class_print(data_site)

split_save_data(data_site, dataset_dir, 'site', TEST_EX, seed=SEED)
