#!/usr/bin/env python
# coding: utf-8

import cv2
import multiprocessing
from multiprocessing import Pool
#multiprocessing.set_start_method('spawn', force=True)

from PIL import Image
import pytesseract

import glob
import numpy as np
import pandas

from mylib.file_preparation import *
from mylib.analysis_rotation import *
from mylib.analysis_computer_vision import *
from mylib.graph_plot import *
from mylib.helpers import *

# 00 create IN-OUT data directories
DATA_DIR = '../DATA_INPUT/'
og_directory = DATA_DIR
work_directory = og_directory+'_TRANS/'

# program results
RESULT_DIR = '../DATA_RESULTS/'
try:
    os.mkdir(RESULT_DIR)
except:
    print(RESULT_DIR, 'already exists')


# 00 make work copy of original
create_data_work_directory(og_directory, work_directory, overwrite=False)

# 0 method
evaluation_methods = ['mlc', 'wer']
measure_method = evaluation_methods[0]
    
''' 2 Resize & Rotate '''
# stores all shape settings [10] plus mlc for each image (6099*10)
SHAPES_PATH = RESULT_DIR+'2_1_all_shapes.csv'

# 2 store best shape
BEST_SHAPES_PATH = RESULT_DIR+'2_2_best_shapes.csv'

# 2 stores best rotation for each resized image (6099)
SHAPE_ROTATION_RESULTS_PATH = RESULT_DIR+'2_3_rotation_results_resize.csv'

# 2 avoid doing rotation again if file exists (6099)
APPLY_SHAPE_ROTATION_RESULTS_PATH = RESULT_DIR+'2_4_apply_rotation_results.csv'

''' 3 Gridsearch'''
GS_WER_RESULTS_PATH = RESULT_DIR+'3_gs_results_wer.csv'
GS_MLC_RESULTS_PATH = RESULT_DIR+'3_gs_results_mlc.csv'
if measure_method is 'wer':
    GS_PATH = GS_WER_RESULTS_PATH
elif measure_method is 'mlc': 
    GS_PATH = GS_MLC_RESULTS_PATH

''' 4 Export '''
#tess_config = "-c tessedit_char_whitelist='1234567890 abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ `´àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ °!\"§$%&/()=?`´#*+-.,ÜÖÄüöä£ß\'"
tess_config = "--oem 1 -c tessedit_char_blacklist='™©®œ}{<>Œ…~¥[]\\@ı_|æ»¢€«Æ>»«' "
EXPORT_RESULTS_PATH = RESULT_DIR+'4_text_export.csv'


''' CPU and batchsize '''
REAL_CPU = multiprocessing.cpu_count()
if REAL_CPU == 1:
    N_CPU = 1
    batch_size = 1
elif REAL_CPU > 1:
    N_CPU = int(REAL_CPU*0.9)
    batch_size = int(N_CPU*0.7) + 1
print('CPU', N_CPU, 'batch-size', batch_size)




print('---------- 0. TRANSFORM ----')
# determine available file types
df_files = get_valid_files(work_directory, filter_extensions=None, filter_trash=True)
df_files.extension.value_counts()

# transform RAW
TRANSFORM_FILE_RAW = RESULT_DIR+'/0_transform_raw.csv'
pipeline_transform_raw(df_files, TRANSFORM_FILE_RAW, N_CPU)

# transform PDF
TRANSFORM_FILE_PDF = RESULT_DIR+'/0_transform_pdf.csv'
pipeline_split_pdf(df_files, TRANSFORM_FILE_PDF, N_CPU)




# transform png
# > 1. grayscaling
print('---------- 1. GRAYSCALE ----')
# 00 as list and dict
exts = ['.png', '.jfif', '.jpeg', '.tiff', '.jpg']
df_files = get_valid_files(work_directory, filter_extensions=exts, filter_trash=True)
valid_files_dict = valid_files_df_to_dict(df_files)
valid_files_list = list(valid_files_dict.keys())

GRAYSCALE_RESULTS_PATH = RESULT_DIR+'/0_grayscale.csv'
grayscale_valid_files(valid_files_list, N_CPU, GRAYSCALE_RESULTS_PATH)




print('---------- 2. ROTATION -----')
df_files = get_valid_files(work_directory, filter_extensions=['.jpg'], filter_trash=True)
valid_files_dict = valid_files_df_to_dict(df_files)
valid_files_list = list(valid_files_dict.keys())

# > 2.1 determine all shapes
shapes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
image_shape_determination(valid_files_list, shapes, N_CPU, batch_size, SHAPES_PATH)

# > 2.2 group by best resize
export_best_shapes(SHAPES_PATH, BEST_SHAPES_PATH)

# > 2.3 determine best rotation 
shape_rotation_determination(BEST_SHAPES_PATH, SHAPE_ROTATION_RESULTS_PATH, N_CPU, batch_size)

# > 2.4 apply rotation correction on best scaled images
apply_rotation_correction(SHAPE_ROTATION_RESULTS_PATH, APPLY_SHAPE_ROTATION_RESULTS_PATH)




# > 3. binarization - use MLC or WER
print('---------- 3. GRIDSEARCH ---')
MLC_THRESHOLD = 38
DEV = True
if not DEV:
    cv_adaptive_sizes = [65,75,85,95,105,115,125,135]
    cv_adaptive_cs = [35,40,45,50]
    cv_adaptive_methods = [0, 1]
else:
    print("GRIDSEARCH - DEV MODE")
    cv_adaptive_sizes = [65,85]
    cv_adaptive_cs = [25,35]
    cv_adaptive_methods = [0, 1]    
    
filtered_valid_files_list = pipeline_filter_quality(SHAPE_ROTATION_RESULTS_PATH, MLC_THRESHOLD)
pipeline_binarization_gridsearch(filtered_valid_files_list, measure_method, cv_adaptive_sizes, cv_adaptive_cs, cv_adaptive_methods, GS_PATH, N_CPU, batch_size)




# > 4. export postcard data as .csv 
print('---------- 4. EXPORT -------')
limit = 20
pipeline_ocr_export(GS_PATH, EXPORT_RESULTS_PATH, tess_config, batch_size, limit, use_binarize=True)




# > 5 hocr export
print('---------- 5. HOCR ---------')
HOCR = True
if HOCR:
    ''' create hocr dir '''
    HOCR_DIR = RESULT_DIR+'HOCR/'
    try:
        os.mkdir(HOCR_DIR)
    except:
        print(HOCR_DIR, 'already exists')

limit = 30
HOCR_RESULTS_PATH = RESULT_DIR+'5_hocr_export.csv'
extract_for_hocs_showcase(GS_PATH, HOCR_RESULTS_PATH, HOCR_DIR, tess_config, limit, N_CPU, batch_size)




# > 6 json export
print('---------- 6. JSON ---------')    
JSON = True
if JSON:
    ''' create hocr dir '''
    JSON_DIR = RESULT_DIR+'JSON/'
    try:
        os.mkdir(JSON_DIR)
    except:
        print(JSON_DIR, 'already exists')
               
results_json_split(EXPORT_RESULTS_PATH, JSON_DIR)




# > 7 pdf export
print('---------- 7. PDF ---------')    
PDF = True
if PDF:
    ''' create hocr dir '''
    PDF_DIR = RESULT_DIR+'PDF/'
    try:
        os.mkdir(PDF_DIR)
    except:
        print(PDF_DIR, 'already exists')

EXPORT_PDF_EXPORT_PATH = RESULT_DIR+'7_pdf_export.csv'
pipeline_export_all_to_pdf(GS_PATH, EXPORT_PDF_EXPORT_PATH, PDF_DIR, tess_config, batch_size)