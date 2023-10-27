# Write by Rodolphe Nonclercq
# July 2023
# ILLS-LIVIA
# contact : rnonclercq@gmail.com


import pandas as pd
import torchvision
from tqdm import tqdm


"""
 This function simplifies annotation and verifies existance of images 
    csv_path : the input annotation path .csv
    save_path : the output annotation path .csv
    img_path : the input image path
"""
def gen_dataframe(csv_path,save_path,img_path):
    dataframe = pd.read_csv(csv_path)
    for elem in ['facial_landmarks']:
        dataframe.pop(elem)
    lost = []
    loop = tqdm(list(dataframe.index))
    for i in loop:
        try:
            torchvision.io.read_image(img_path+'/'+dataframe.loc[i,'subDirectory_filePath'])
            lost.append(0)
        except:
            lost.append(1)
    dataframe['lost'] = lost
    dataframe.to_csv(save_path)

# Example
if __name__ == '__main__':
    save_path = '.../AffectNet/training_wo_landmark.csv'
    csv_path = '.../AffectNet/training.csv'
    img_path = '.../AffectNet/Manually_Annotated_Images'
    gen_dataframe(csv_path,save_path,img_path)