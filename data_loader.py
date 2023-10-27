# Write by Rodolphe Nonclercq
# June 2023
# ILLS-LIVIA
# contact : rnonclercq@gmail.com


import os,glob
import pandas as pd
import torch
import torchvision
import random
import numpy as np
from tqdm import tqdm


class Dataset_MOVI_image(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,transform = lambda x: x,nb_image=None,crop=False):
        self.PATH_IMG = PATH_IMG
        self.transform = transform
        self.nb_image = nb_image
        self.dataframe = pd.DataFrame.from_dict({'name':os.listdir(PATH_IMG)})
        self.dataframe['img_id'] = 0

        self.reset()
        
    def reset(self):
        self.dataframe['img_id'] = np.random.randint(24,size=(len(self.dataframe),))
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image).index)
        else:
            self.index_link = list(self.dataframe.index)
    
    def __len__(self):
        return len(self.index_link)
    
    def __getitem__(self,idx):
        index = self.index_link[idx]
        img_tensor = torchvision.io.read_image(self.PATH_IMG+'/'+self.dataframe.loc[index,'name']+'/'+str(self.dataframe.loc[index,'img_id']).zfill(8)+'_image.png')
        img_tensor = self.transform(img_tensor)
        return img_tensor
    
class Dataset_AffectNet_image(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,nb_image=None,type_output='expression',crop=False,balanced=True,nb_class=8,preload=False,loaded_resolution=224):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.type_output = type_output
        self.crop = crop
        self.balanced = balanced
        self.nb_class = nb_class
        self.reset()
        self.preloaded = preload
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        dataframe = pd.read_csv(self.PATH_ANOT,index_col=0)
        dataframe = dataframe[(dataframe['lost'] == 0) & (dataframe['expression'] < self.nb_class)]
        N = len(dataframe)
        self.dic_image = {}
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        if preload:
            loader = tqdm(dataframe.index)
            for i,p in enumerate(loader):
                img = torchvision.io.read_image(self.PATH_IMG+'/'+dataframe.loc[p,'subDirectory_filePath'])
                self.dic_image[p] = i
                if self.crop:
                    img = torchvision.transforms.functional.crop(img,
                                                            int(dataframe.loc[p,'face_x']),
                                                            int(dataframe.loc[p,'face_y']),
                                                            int(dataframe.loc[p,'face_height']),
                                                            int(dataframe.loc[p,'face_width']))
                img = self.resize(img)
                self.img_loaded[i]=img
        
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self):
        dataframe = pd.read_csv(self.PATH_ANOT,index_col=0)
        self.dataframe = dataframe[(dataframe['lost'] == 0) & (dataframe['expression'] < self.nb_class)]
        if self.nb_image is None:
            if self.balanced:
                nb_image_per_class = None
                nb_class = len(set(self.dataframe['expression']))
                for exp_class in range(nb_class):
                    size_class=  len(self.dataframe[self.dataframe['expression'] == exp_class])
                    if nb_image_per_class is None or nb_image_per_class > size_class:
                        nb_image_per_class = size_class
                    self.nb_image = nb_image_per_class*nb_class
                    
            else:
                self.nb_image = len(self.dataframe)
                
            
        if self.balanced:
            self.dataframe.loc[:,['usage']] = 0
            nb_class = len(set(self.dataframe['expression']))
            nb_image_per_class = self.nb_image//nb_class
            self.nb_image = nb_image_per_class*nb_class
            for expression_class in range(nb_class):
                indexes = list(self.dataframe[self.dataframe['expression'] == expression_class].sample(nb_image_per_class).index)
                self.dataframe.loc[indexes,['usage']] = 1
            
            self.dataframe = self.dataframe[self.dataframe['usage'] > 0]

        
        self.index_link = list(self.dataframe.sample(self.nb_image).index)
        self.index_link.sort()

    def __getitem__(self,idx):
        index = self.index_link[idx]

        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            l_img = torchvision.io.read_image(self.PATH_IMG+'/'+self.dataframe.loc[index,'subDirectory_filePath'])
            l_img = torchvision.transforms.functional.crop(l_img,
                                                            int(self.dataframe.loc[index,'face_x']),
                                                            int(self.dataframe.loc[index,'face_y']),
                                                            int(self.dataframe.loc[index,'face_height']),
                                                            int(self.dataframe.loc[index,'face_width']))
            self.img_loaded[i] = self.resize(l_img)
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        if self.type_output == 'expression':
            target_tensor = self.dataframe.loc[index,'expression']
        elif self.type_output == 'VA':
            target_tensor = torch.zeros((2,))
            target_tensor[0] = self.dataframe.loc[index,'valence']
            target_tensor[1] = self.dataframe.loc[index,'arousal']
        
        return img_tensor,target_tensor


class Dataset_UNBC_image(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,balanced=None,IDs = None,img_per_class = 2500):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.img_per_class = img_per_class
        self.balanced = balanced
        self.IDs = IDs
        self.reset()
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self):
        dataframe = pd.read_csv(self.PATH_ANOT,index_col=0)

        if self.IDs is None:
            self.IDs = set(dataframe['ID'])
        
        self.dataframe = dataframe[(dataframe['lost'] == 0) & (dataframe['ID'].isin( self.IDs))].copy()
        if not self.balanced is None:
            self.dataframe.loc[:,'usage'] = 0
            self.nb_pain_level = len(set(self.dataframe['pain']))
            for pain in set(self.dataframe['pain']):
                size = len(self.dataframe[self.dataframe['pain'] == pain])

                if self.balanced == 'resample':

                    for it in range(self.img_per_class//size):
                        self.dataframe.loc[self.dataframe['pain'] == pain,'usage'] += 1
                    indexes = list(self.dataframe[self.dataframe['pain'] == pain].sample(self.img_per_class%size).index)
                    self.dataframe.loc[indexes,['usage']] += 1

                elif self.balanced == 'threshold':

                    if len(self.dataframe[self.dataframe['pain'] == pain]) > self.img_per_class:
                        indexes = list(self.dataframe[self.dataframe['pain'] == pain].sample(self.img_per_class).index)
                        self.dataframe.loc[indexes,['usage']] += 1
                    else:
                        indexes = list(self.dataframe[self.dataframe['pain'] == pain].index)
                        self.dataframe.loc[indexes,['usage']] += 1
            
            self.index_link = []
            for u in range(1,max(self.dataframe['usage'])+1):
                self.index_link += u*list(self.dataframe[self.dataframe['usage'] == u].index)
        else:
            self.index_link = list(self.dataframe.index)

    def weight(self):
        size = len(set(self.dataframe['pain']))
        weight = torch.zeros((size,))
        for ind in self.index_link:
            weight[self.dataframe.loc[ind,'pain']] += 1
        return len(self.index_link)/size/weight

    def __getitem__(self,idx):
        index = self.index_link[idx]
        img_tensor = self.transform(torchvision.io.read_image(self.PATH_IMG+'/'+self.dataframe.loc[index,'path']))
        pain_tensor = self.dataframe.loc[index,'pain']
        ID_tensor = self.dataframe.loc[index,'ID']
        return img_tensor,pain_tensor,ID_tensor

class Dataset_AffWild_image(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.reset()
        
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self):
        dataframe = pd.read_csv(self.PATH_ANOT,index_col=0)

        if self.IDs is None:
            self.IDs = set(dataframe['ID'])
        if set is None:
            self.dataframe = dataframe[(dataframe['lost'] == 0) & (dataframe['ID'].isin( self.IDs))].copy()
        else:
            self.dataframe = dataframe[(dataframe['lost'] == 0) & (dataframe['set'] == self.set_type) & (dataframe['ID'].isin( self.IDs))].copy()

        if not self.nb_image is None:
            self.dataframe.loc[:,'usage'] = 0
            indexes = list(self.dataframe.sample(self.nb_image).index)
            self.dataframe.loc[indexes,['usage']] += 1
            self.index_link = list(self.dataframe[self.dataframe['usage'] == 1].index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        img_tensor = self.transform(torchvision.io.read_image(self.PATH_IMG+'/'+self.dataframe.loc[index,'path']))
        VA_tensor = torch.zeros((2,))
        VA_tensor[0] = self.dataframe.loc[index,'valence']
        VA_tensor[1] = self.dataframe.loc[index,'arousal']
        ID_tensor = self.dataframe.loc[index,'ID']
        return img_tensor,VA_tensor, ID_tensor

class Dataset_AffWild_video(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,set_type, size_video = 128, transform = lambda x: x, nb_video = None,init_frequency = 1):
        self.video_path = glob.glob(PATH_ANOT+'/'+set_type+'/*.csv')
        self.dataframe = pd.DataFrame.from_dict({'path':self.video_path})
        self.PATH_ANOT = PATH_ANOT
        self.PATH_IMG = PATH_IMG
        self.nb_video = nb_video
        self.size_video = size_video
        self.transform = transform
        self.dataframe['seed'] = 0
        self.dataframe['frequency'] = init_frequency
        self.reset()
    def reset(self):
        if not self.nb_video is None:
            self.dataframe.loc[:,'usage'] = 0
            indexes = list(self.dataframe.sample(self.nb_video).index)
            self.dataframe.loc[indexes,['usage']] = 1
            self.dataframe.loc[indexes,['seed']] = np.random.randint(1000,size=(len(indexes),))
            self.dataframe.loc[indexes,['frequency']] = np.random.randint(4,10,size=(len(indexes),))
            self.index_link = indexes
        else:
            indexes = list(self.dataframe.index)
            self.dataframe.loc[indexes,['usage']] = 1
            self.dataframe.loc[indexes,['seed']] = np.random.randint(1000,size=(len(indexes),))
            self.dataframe.loc[indexes,['frequency']] = np.random.randint(4,10,size=(len(indexes),))
            self.index_link = indexes
        
    def __len__(self):
        return len(self.index_link)
    
    def __getitem__(self,idx):
        index = self.index_link[idx]
        seed = self.dataframe.loc[index,'seed']
        freq = self.dataframe.loc[index,'frequency']
        dataframe_video =  pd.read_csv(self.dataframe.loc[index,'path'],index_col=0)
        d = dataframe_video.iloc[:-freq*self.size_video+1][dataframe_video.iloc[:-freq*self.size_video+1]['group'] != -1]
        idx_image = list(d.sample(1,random_state=seed).index)[0]
        x = torch.zeros((self.size_video,3,112,112),dtype=torch.uint8)
        VA_tensor = torch.zeros(self.size_video,2)
        for i in range(self.size_video):
            if dataframe_video.loc[idx_image+freq*i,'group'] != -1:
                x[i] = torchvision.io.read_image(self.PATH_IMG+'/'+dataframe_video.loc[idx_image+freq*i,'img'])
                VA_tensor[i,0] = dataframe_video.loc[idx_image+freq*i,'V']
                VA_tensor[i,1] = dataframe_video.loc[idx_image+freq*i,'A']
            else:
                x[i] = x[i-1]
                VA_tensor[i] = VA_tensor[i-1]
                
        return self.transform(x), VA_tensor

class Dataset_Biovid_image_binary_class(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None,nb_fold=1,preload=False,loaded_resolution=224):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.preload = preload
        self.reset()
        q_fold = 40//nb_fold
        self.fold = [[j for j in range(i*q_fold,(i+1)*q_fold)] for i in range(nb_fold)]

        dataframe = pd.read_csv(self.PATH_ANOT)
        N = len(dataframe)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image = {}
        if preload:
            loader = tqdm(dataframe.index)
            for i,p in enumerate(loader):
                img = torchvision.io.read_image(self.PATH_IMG+'/'+self.dataframe.loc[p,'path'][20:])
                self.dic_image[p] = i
                img = self.resize(img)
                self.img_loaded[i]=img
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self,fold=None,keep=False):
        dataframe = pd.read_csv(self.PATH_ANOT)
        self.dataframe = dataframe
        if not fold is None and fold < len(self.fold) and fold >=0:
            if not keep:
                self.dataframe = self.dataframe[~ self.dataframe['id_video'].isin(self.fold[fold])]
            else:
                self.dataframe = self.dataframe[self.dataframe['id_video'].isin(self.fold[fold])]

        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe['ID']))}
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            self.img_loaded[i] = self.resize(torchvision.io.read_image(self.PATH_IMG+'/'+self.dataframe.loc[index,'path'][20:]))
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        pain_tensor = self.dataframe.loc[index,'pain']
        ID_tensor = self.dic_ID[self.dataframe.loc[index,'ID']]
        
        return img_tensor, pain_tensor,ID_tensor

class Dataset_Biovid_image(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.reset()
        
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self):
        dataframe = pd.read_csv(self.PATH_ANOT)

        self.dataframe = dataframe
        self.dataframe['seed'] = np.random.randint(1000,size=(len(self.dataframe),))
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        path = self.PATH_IMG+'/'+self.dataframe.loc[index,'path']+'/*'
        glob_path = glob.glob(path)
        random.seed(int(self.dataframe.loc[index,'seed']))
        try:
            path = glob_path[random.randint(0,len(glob_path)-1)]
        except:
            print(path)
        
        img_tensor = self.transform(torchvision.io.read_image(path))
        pain_tensor = self.dataframe.loc[index,'pain']
        return img_tensor, pain_tensor

#%%
#Example Dataset_Biovid_image_binary_class
if __name__ == '__main__': 
    Biovid_img = '.../Biovid/sub_red_classes_img'

    # Check /Dataset processing/Biovid.py for train and test set
    biovid_annot_train = '.../Biovid/binary/train.csv'
    biovid_annot_test = '.../Biovid/binary/test.csv'

    BATCH_SIZE = 200
    RESOLUTION = 112
    nb_class = 8
    nb_ID = 61
    FOLD = 8

    tr = data_augm(RESOLUTION)
    tr_test = data_adapt(RESOLUTION)
    tr_size = torchvision.transforms.Resize((RESOLUTION,RESOLUTION),antialias=True)

    dataset_train = Dataset_Biovid_image_binary_class(Biovid_img,biovid_annot_train,transform = tr.transform,IDs = None,nb_image = None,nb_fold=FOLD,preload=True)



    loader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=BATCH_SIZE, shuffle=True,
                                                num_workers=4,drop_last = True)

    dataset_test = Dataset_Biovid_image_binary_class(Biovid_img,biovid_annot_test,transform = tr_test.transform,IDs = None,nb_image = None,preload=True)




    loader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=BATCH_SIZE, shuffle=True,
                                                num_workers=4)

#%%
#Example Dataset_Biovid_image_binary_class
if __name__ == '__main__':
    BATCH_SIZE = 100
    RESOLUTION = 224
    nb_class = 7

    # Check /Dataset processing/AffectNet.py 
    affectNet_img = '.../AffectNet/Manually_Annotated_Images'
    affectNet_annot_train = '.../AffectNet/training_wo_landmark.csv'
    affectNet_annot_val = '.../AffectNet/validation_wo_landmark.csv'

    tr = data_augm(RESOLUTION)
    tr_test = data_adapt(RESOLUTION)
    tr_size = torchvision.transforms.Resize((RESOLUTION,RESOLUTION),antialias=True)

    dataset_train = Dataset_AffectNet_image(affectNet_img, affectNet_annot_train,
                                    tr.transform,crop=True,nb_image=None,
                                    type_output='expression',balanced=True,nb_class=nb_class,preload=False)

    dataset_test = Dataset_AffectNet_image(affectNet_img, affectNet_annot_val,
                                    tr_test.transform,crop=True,type_output='expression',
                                    balanced=False,nb_class=nb_class,preload=False)
    
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=4,drop_last = True)

    loader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=4)
