import SimpleITK as sitk
import numpy as np
import os
import random
import pandas as pd

def normalize(arr):
    return (arr-np.mean(arr)) / np.std(arr)

random.seed(10)


info_path = '/home/lsy/PMC-Fed/dataset/cervical/dataset_9PartyID.csv'

all_df = pd.read_csv(info_path)

id_index_dict = {}

for index, row in all_df.iterrows():
    id_value = row['ID']
    if id_value in id_index_dict:
        id_index_dict[id_value].append(index)
    else:
        id_index_dict[id_value] = [index]

###以7:1.5:1.5的比例划分训练测试验证集
train_index = []
val_index = []
test_index = []

for key in id_index_dict:
    index_list = id_index_dict[key]
    random.shuffle(index_list)
    train_index += index_list[:int(len(index_list)*0.7)]
    val_index += index_list[int(len(index_list)*0.7):int(len(index_list)*0.85)]
    test_index += index_list[int(len(index_list)*0.85):]

assert len(train_index) + len(val_index) + len(test_index) == len(all_df)

all_df['split'] = 'train'
all_df.loc[val_index, 'split'] = 'val'
all_df.loc[test_index, 'split'] = 'test'

all_df.to_csv('/home/lsy/PMC-Fed/dataset/cervical/dataset_9PartyID_new.csv', index=False)