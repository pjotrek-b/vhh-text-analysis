# +
import shutil, os, io
from PIL import Image

import numpy as np
import csv
import pandas as pd


# +
def check_if_file_in_list(file_path, gs_list):
    if os.path.exists(file_path):
        if file_path not in gs_list:
            return True
        else:
            return False
    else:
        print('Error: file does not exist or invalid path')
        
def get_new_files_to_be_processed(path, col_names, index_col, all_files):
    '''
    path (str) - to datafile
    col_names (list) - of datafile if not existent
    index_col (str) - unique identifier 
    all_files (list) - list of all files
    RETURN files that are not processed yet
    '''
    if os.path.exists(path):
        already_processed_files = pd.read_csv(path, header=0)
        already_processed_files = list(already_processed_files[index_col])
        all_files = all_files
        new_files = list(set(all_files) - set(already_processed_files))
        print('load existing data..')
        print('new:', len(new_files), 'total:',len(all_files), 'existing:', len(already_processed_files))
    else: 
        with open(path, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(col_names)
        new_files = all_files
        print('create new data..')
        print('new:', len(new_files), 'total:',len(all_files))
    return new_files


# +

def job(path, arg_2, arg_3, arg_4):
    print(arg_2[0], arg_3[0], arg_4)

    img_bin_adapt = cv2.imread(path, 0)
    if arg_4[0] == 0:
        print('y')
        img_bin_adapt = cv2.adaptiveThreshold(img_bin_adapt,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY, 15, 15)
        return img_bin_adapt
    elif arg_4[0] == 1:
        print('yyy')
        img_bin_adapt = cv2.adaptiveThreshold(img_bin_adapt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 15, 15)
        return img_bin_adapt
        
    return path
# -


