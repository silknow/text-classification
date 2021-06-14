# coding: utf-8

import os
from ihelper import read_i_json, read_label_map, convert_labels
from ihelper import ex_p_class_print, ex_p_class_min, split_save_data
from ihelper import remove_short_texts, remove_duplicates
from ihelper import count_unique_labels, delete_data

# Metadata Imatex2

# keys of the desired information in the json-file
key_epoch_i = 'CRONOLOGIA*'
key_production_site_i = 'ORIGEN*'
key_material_i = 'MATÈRIES*'
key_method_i = 'TÈCNICA*'
key_subject_i = 'DECORACIÓ*'
key_txt = 'DESCRIPCIÓ'

# paths to the images and the labels
# path_imatex2_images = r''
data_dir = r'../data/imatex/records'
collection_path = '../TrainingSamples'
file_sufix = '*_ca.json'
temp_dir = '../temp'
dataset_dir = '../datasets/imatex'

# Parameters
SEED = 201906
MIN_TOKENS = 5
MIN_EX = 50   # minimum number of examples per class: train + test
TEST_EX = 0.4  # test set examples (percentage)

#
# Get information out of the records
#
data_epoch = read_i_json(data_dir, file_sufix, key_txt, key_epoch_i)
data_material = read_i_json(data_dir, file_sufix, key_txt, key_material_i)
data_method = read_i_json(data_dir, file_sufix, key_txt, key_method_i)
data_site = read_i_json(data_dir, file_sufix, key_txt, key_production_site_i)
data_subject = read_i_json(data_dir, file_sufix, key_txt, key_subject_i)

print('Examples Read:')
print('\tEpoch: {}'.format(len(data_epoch)))
print('\tMaterial: {}'.format(len(data_material)))
print('\tMethod: {}'.format(len(data_method)))
print('\tProduction Site: {}'.format(len(data_site)))
print('\tSubject: {}'.format(len(data_subject)))
print('Done reading json files')

#
# Read label map files
#
p = os.path.join(collection_path, '_epoch_descriptions2label_dat_labeled2.txt')
epoch_label_map = read_label_map(p)

p = os.path.join(collection_path,
                 '_material_descriptions2label_dat_labeled.txt')
material_label_map = read_label_map(p)

p = os.path.join(collection_path, '_method_descriptions2label_dat_labeled.txt')
method_label_map = read_label_map(p)

p = os.path.join(collection_path, '_site_descriptions2label_dat_labeled2.txt')
site_label_map = read_label_map(p)

p = os.path.join(collection_path,
                 '_subject_descriptions2label_dat_labeled.txt')
subject_label_map = read_label_map(p)

print('Label mappings Read:')
print('\tEpoch: {}'.format(len(epoch_label_map)))
print('\tMaterial: {}'.format(len(material_label_map)))
print('\tMethod: {}'.format(len(method_label_map)))
print('\tProduction Site: {}'.format(len(site_label_map)))
print('\tSubject: {}'.format(len(subject_label_map)))
print('Done reading label maps')


#
# Preprocessing: convertions, duplicates, short desc
#
# Remove short descriptions
data_epoch = remove_short_texts(data_epoch, MIN_TOKENS)
data_material = remove_short_texts(data_material, MIN_TOKENS)
data_method = remove_short_texts(data_method, MIN_TOKENS)
data_site = remove_short_texts(data_site, MIN_TOKENS)
data_subject = remove_short_texts(data_subject, MIN_TOKENS)

# Convert record labels
data_epoch, err_epoch = convert_labels(data_epoch, epoch_label_map)
data_material, err_material = convert_labels(data_material, material_label_map)
data_method, err_method = convert_labels(data_method, method_label_map)
data_site, err_site = convert_labels(data_site, site_label_map)
data_subject, err_subject = convert_labels(data_subject, subject_label_map)

print('Examples Converted:')
print('\tEpoch: {}'.format(len(data_epoch)))
print('\tMaterial: {}'.format(len(data_material)))
print('\tMethod: {}'.format(len(data_method)))
print('\tProduction Site: {}'.format(len(data_site)))
print('\tSubject: {}'.format(len(data_subject)))
print('Done converting records')

# Remove duplicates
data_epoch = remove_duplicates(data_epoch)
data_material = remove_duplicates(data_material)
data_method = remove_duplicates(data_method)
data_site = remove_duplicates(data_site)
data_subject = remove_duplicates(data_subject)

#
# Errors and Stats
# Write errors
p = os.path.join(temp_dir, 'missing_epoch.txt')
with open(p, 'w') as fout:
    for bad in err_epoch:
        print('{}'.format(bad), file=fout)
p = os.path.join(temp_dir, 'missing_material.txt')
with open(p, 'w') as fout:
    for bad in err_material:
        print('{}'.format(bad), file=fout)
p = os.path.join(temp_dir, 'missing_method.txt')
with open(p, 'w') as fout:
    for bad in err_method:
        print('{}'.format(bad), file=fout)
p = os.path.join(temp_dir, 'missing_site.txt')
with open(p, 'w') as fout:
    for bad in err_site:
        print('{}'.format(bad), file=fout)
p = os.path.join(temp_dir, 'missing_subject.txt')
with open(p, 'w') as fout:
    for bad in err_subject:
        print('{}'.format(bad), file=fout)
print('Done writing Errors')

# Stats
ex_p_class_print(data_epoch, 'stats_epoch.txt', dst_dir=temp_dir)
ex_p_class_print(data_material, 'stats_material.txt', dst_dir=temp_dir)
ex_p_class_print(data_method, 'stats_method.txt', dst_dir=temp_dir)
ex_p_class_print(data_site, 'stats_site.txt', dst_dir=temp_dir)
ex_p_class_print(data_subject, 'stats_subject.txt', dst_dir=temp_dir)
print('Done printing class stats')


#
# Create training and test sets
# - Here we should change the language of the label at least but hey
#
# min number of examples per class
data_epoch = ex_p_class_min(data_epoch, MIN_EX)
data_material = ex_p_class_min(data_material, MIN_EX)
data_method = ex_p_class_min(data_method, MIN_EX)
data_site = ex_p_class_min(data_site, MIN_EX)
data_subject = ex_p_class_min(data_subject, MIN_EX)
print('Done filtering for min num examples')

# delete files from previous run
# ignore tasks that are not viable
if count_unique_labels(data_epoch) < 2:
    data_epoch = None
    delete_data(dataset_dir, 'epoch')
if count_unique_labels(data_material) < 2:
    data_material = None
    delete_data(dataset_dir, 'material')
if count_unique_labels(data_method) < 2:
    data_method = None
    delete_data(dataset_dir, 'method')
if count_unique_labels(data_site) < 2:
    data_site = None
    delete_data(dataset_dir, 'site')
if count_unique_labels(data_subject) < 2:
    data_subject = None
    delete_data(dataset_dir, 'subject')

# Stats
print('-' * 80)
print('epoch')
ex_p_class_print(data_epoch)
print('\nmaterial')
ex_p_class_print(data_material)
print('\nmethod')
ex_p_class_print(data_method)
print('\nsite')
ex_p_class_print(data_site)
print('\nsubject')
ex_p_class_print(data_subject)
print('-' * 80)

# split train/test and save
split_save_data(data_epoch, dataset_dir, 'epoch', TEST_EX,  seed=SEED)
split_save_data(data_material, dataset_dir, 'material', TEST_EX, seed=SEED)
split_save_data(data_method, dataset_dir, 'method', TEST_EX, seed=SEED)
split_save_data(data_site, dataset_dir, 'site', TEST_EX, seed=SEED)
split_save_data(data_subject, dataset_dir, 'subject', TEST_EX, seed=SEED)
print('Done saving dataset')
