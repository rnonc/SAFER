# Write by Rodolphe Nonclercq
# October 2023
# ILLS-LIVIA
# contact : rnonclercq@gmail.com


import pandas as pd


"""
 This function extracts ID and video ID + splits the official train set and the official test set
    text_file : the input path .txt
    save_path : the output path .csv
    select_name_valid : List of official names in the test (the rest go in the train) if select_name_valid = None gen_dataframe just convert .txt to .csv
"""
def gen_dataframe(text_file,save_path,select_name_valid=None):
    dataframe = pd.read_csv(text_file, sep=" ", header=None)
    dataframe = dataframe.rename({0:'path',1:'pain'},axis='columns')
    dataframe.loc[:,['ID','id_video']] = -1
    if not select_name_valid is None:
        id_name_file = dataframe['path'].str.split('/',expand=True)[1]
        set_name = set(id_name_file)
        for i,name in enumerate(set_name):
            dataframe.loc[id_name_file[id_name_file == name].index,['ID']] = i

            name_video = dataframe[id_name_file == name]['path'].str.split('/',expand=True)[2]
            set_video = set(name_video)
            for j,video in enumerate(set_video):
                dataframe.loc[name_video[name_video == video].index,['id_video']] = j
        valid_lines = id_name_file.isin(select_name_valid)
        dataframe_test  = dataframe[valid_lines]
        dataframe_train  = dataframe[valid_lines == False]
        dataframe_train.to_csv(save_path+'/train.csv',index=False)
        dataframe_test.to_csv(save_path+'/test.csv',index=False)
    else:
        dataframe.to_csv(save_path+'/train.csv')


# Example
if __name__ == "__main__":
    gen_dataframe('.../Biovid/sub_two_labels.txt','.../Biovid/binary',['100914_m_39','101114_w_37',
                                                                        '082315_w_60', '083114_w_55',
                                                                        '083109_m_60','072514_m_27',
                                                                        '080309_m_29', '112016_m_25',
                                                                        '112310_m_20', '092813_w_24',
                                                                        '112809_w_23', '112909_w_20',
                                                                        '071313_m_41', '101309_m_48',
                                                                        '101609_m_36', '091809_w_43',
                                                                        '102214_w_36', '102316_w_50',
                                                                        '112009_w_43', '101814_m_58',
                                                                        '101908_m_61', '102309_m_61',
                                                                        '112209_m_51', '112610_w_60',
                                                                        '112914_w_51', '120514_w_56'])