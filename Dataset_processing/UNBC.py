import os,glob
import pandas as pd
import torch
import torchvision
import time
import random
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    annotation_UNBC = '/home/ens/AT91140/project_DA/Datasets/UNBC-McMaster/list_full.txt'
    cropp_face_UNBC = '/home/ens/AT91140/project_DA/Datasets/UNBC-McMaster/Aligned-Images'
    os.chdir('/home/ens/AT91140/project_DA/Datasets/UNBC-McMaster')
    dataframe = pd.read_csv(annotation_UNBC, sep=" ", header=None)
    dataframe = dataframe.rename({0:'path',
                                  1:'pain'},axis='columns').drop([2,3],axis=1)
    dataframe['lost'] = -1
    dataframe['ID'] = -1

    current_id = 0
    current_id_name = dataframe['path'].iloc[0].split('/')[0]
    loop = tqdm(range(len(dataframe)))
    for i in loop:
        if dataframe['path'].iloc[i].split('/')[0] == current_id_name:
            dataframe.at[i,'ID'] = current_id
        else:
            current_id_name = dataframe['path'].iloc[i].split('/')[0]
            current_id += 1
            dataframe.at[i,'ID']  = current_id
        try:    
            torchvision.io.read_image(cropp_face_UNBC+'/'+dataframe['path'].iloc[i])
            dataframe.at[i,'lost']  = 0
        except:
            dataframe.at[i,'lost']  = 1
    
    dataframe.to_csv('pain.csv')
