import SimpleITK as sitk
import numpy as np
import os
import random

def normalize(arr):
    return (arr-np.mean(arr)) / np.std(arr)

random.seed(10)
clients = ['RUNMC', 'BMC', 'I2CVB', 'UCL','BIDMC', 'HK']

#path to the original data
base_path = '/data/lsy/ProstateMRI/'

#path to save the processed data
tar_path = '/data/lsy/ProstateMRI/processed/'

#path to save the info of the processed data
info_path = './dataset/Prostate'

####choose 3 channels or 1 channel
channel = 3

if not os.path.exists(info_path): 
    os.makedirs(info_path)

for client in clients:
    folder = os.path.join(base_path, client)
    nii_seg_list = [nii for nii in os.listdir(folder) if 'segmentation' in str(nii).lower()]

    tar_folder_path = os.path.join(tar_path,client)
    if not os.path.exists(tar_folder_path): 
        os.makedirs(tar_folder_path)
    
    slices_list = []
    for nii_seg in nii_seg_list:
        nii_path = os.path.join(folder, nii_seg[:6]+'.nii.gz')
        nii_seg_path = os.path.join(folder, nii_seg)

        image_vol = sitk.ReadImage(nii_path)
        label_vol = sitk.ReadImage(nii_seg_path)
        image_vol = sitk.GetArrayFromImage(image_vol)
        label_vol = sitk.GetArrayFromImage(label_vol)
        label_vol[label_vol > 1] = 1
        has_label = list(set(np.where(label_vol>0)[0]))
        
        label_vol = label_vol[has_label]
        image_vol = image_vol[has_label]


        for i in range(image_vol.shape[0]):
            slice_name = nii_seg[:6] + '_slice_' + str(i)
            slices_list.append(slice_name)
            
            if channel == 3:

            # merge slice into 3 channels (with upper and lower slice)
                if i==0:
                    image = np.concatenate([np.expand_dims(image_vol[0, :, :],0),image_vol[i:i+2, :, :]],axis=0)
                elif i==image_vol.shape[0]-1:
                    image = np.concatenate([image_vol[i-2:i, :, :],np.expand_dims(image_vol[i, :, :],0)])
                else:
                    image = np.array(image_vol[i-1:i+2, :, :])
                    

                image = np.transpose(image,(1,2,0))     
                mask = label_vol[i, :, :]

                assert image.shape == (384, 384,3)
        
            elif channel==1:

                image = image_vol[i, :, :]
                mask = label_vol[i, :, :]

                assert image.shape == (384, 384)
                assert mask.shape == (384, 384)

            np.save(os.path.join(tar_folder_path, slice_name + '.npy'), image)
            np.save(os.path.join(tar_folder_path, slice_name + '_segmentation.npy'), mask)

    slices_num = len(slices_list)
    random.shuffle(slices_list)
    train_num = int(slices_num * 0.7)
    val_num = int(slices_num * 0.15)
    test_num = slices_num - train_num - val_num
    
    train_list = slices_list[:train_num]
    val_list = slices_list[train_num:train_num+val_num]
    test_list = slices_list[train_num+val_num:]

    assert len(train_list) + len(val_list) + len(test_list) == slices_num

    info = {'train': train_list, 'val': val_list, 'test': test_list}
    print('client:{} total:{} train:{} val:{} test:{}'.format(client, len(train_list) + len(val_list) + len(test_list), len(train_list), len(val_list), len(test_list)))
    for split in ['train', 'val', 'test']:
        with open(os.path.join(info_path, client + '_' + split + '.txt'), 'w') as f:
            for slice_name in info[split]:
                f.write(slice_name + '\n')

    


