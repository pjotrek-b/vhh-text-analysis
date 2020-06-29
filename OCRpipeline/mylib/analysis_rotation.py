import numpy as np
import pytesseract
from PIL import Image
import shutil, os, io
import csv
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import cv2

# ### taken from helpers.py

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


# from mylib.file_preparation import *

# ## Rotation
# we tackle multiple/different strategies ..
# - Word Similarity
# - MLC
# - Tesseract Orientation

# +
# taken from file_preparation.py
def save_pil_image(pil_img, to_path):
    try:
        from PIL import Image
        pil_img.save(to_path)
    except:
        print(e, 'could not save image to desired path')

def correct_img_rotation(file, rotation):
    '''
    rotate file and safe replace to original path
    '''
    try:
        rotated_img = rotate_image(file, rotation) 
        save_pil_image(rotated_img, file)
        return True
    except Exception as e:
        print('Err:', e)
        return False


# -

def rotate_image(img_path, angle, expand=True):
    '''
    return rotated PIL image
    '''
    from PIL import Image
    img = Image.open(img_path).rotate(angle, expand=expand)
    return img


# +
def get_osd_info(img_path):
    import pytesseract
    '''
    parse tesseract osd output string to a dict 
    for easier variable reading.
    Info:
        int Page_number
        int Orientation_in_degrees
        int Rotate
        float  Orientation_confidence
        string Script
        float  Script_confidence
    '''
    osd_info = pytesseract.image_to_osd(img_path)
    info = {}
    for i in osd_info.split('\n'):
        try:
            if '.' in i.split(':')[1].strip():
                info[str(i.split(':')[0].replace(' ', '_'))] = float(i.split(':')[1].strip())
            else:
                info[str(i.split(':')[0].replace(' ', '_'))] = int(i.split(':')[1].strip())
        except:
            info[str(i.split(':')[0].replace(' ', '_'))] = i.split(':')[1].strip()
    return info


def build_data_get_osd_info(img_path):
    try:
        info = get_osd_info(img_path)
        info['file'] = img_path
        info['correction'] = 0
        next_res = info.values()
        #print('>> good', img_path)
        return list(next_res)
    except:
        next_res = list(np.full([8-2], np.nan))
        next_res.append(img_path)
        next_res.append(0) 
        #print('>> bad', img_path)
        return list(next_res)


# +
def similarity(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def calc_box_similarities(box_data, sim_threshold=0.8, reference_words=['declassified', 'authority']):
    '''
    return
    Fale if similarity is low - rotation is wrong 
    True if rotation correct because reference words been detected within the defined threshold
    '''
    reference_words = reference_words
    sim_thresh = sim_threshold
    top_references = {}
    box_data_low = [x.lower().strip() for x in box_data]

    for ref in reference_words:
        top_sim = 0
        top_word = ''
        for word in box_data_low:
            if word.strip():
                if similarity(ref, word) > top_sim:
                    top_sim = similarity(ref, word)
                    top_word = word
        top_references[top_word] = top_sim

    for f in list(top_references.items())[:]:
        if (f[1]) > sim_threshold:
            return True, top_references
    
    return False, top_references


# -
def calc_mean_line_confidence(df_data):
    '''
    return mean line confidence of lines that are not NaN or empty
    '''
    import numpy as np
    try:
        if df_data.shape[0] > 1:
            mean_confidence = df_data[(df_data.text.notnull()) &  (df_data.text.str.strip() != '')].conf.mean()
            return mean_confidence
        else:
            return np.nan
    except:
        print('Error: mean line confidence')
        return np.nan

# +
def word_error_rate(ground_truth, hypothesis):
    from jiwer import wer
    error = wer(ground_truth, hypothesis)
    return error

def get_best_rotation(file_path, resize_p=1, output=False, eval_method='mlc', ground_truth=None):
    '''
    1. resize image if resize_p is not 1
    2. rotate image in all 4 directions
    3. evaluate on eval_method = [mlc, wer]
        mean line confidence
        word error rate -> requires ground_truth
    4. return best result
    '''
    
    if eval_method not in ['mlc', 'wer']:
        print('Invalid eval_method')
        return False
    
    if ground_truth is None and eval_method is 'wer':
        print('Please provide a gound truth text if evaluation method == wer')
        return False
    
    if eval_method == 'mlc':
        best_eval = 0
    elif eval_method == 'wer':
        best_eval = 100
    
    import time
    best_rotation = 0
    start = time.time()
    
    for r in [0,90,180,270]:
        img = Image.open(file_path)
        
        new_size = (int(img.size[0]*resize_p), int(img.size[1]*resize_p))
        img = img.resize(new_size)

        rotate_angle = r
        img = img.rotate(rotate_angle, expand=True)

        data = pytesseract.image_to_data(img, output_type='data.frame', lang='eng')
        
        img.close()
        
        if eval_method == 'mlc':
            mlc = calc_mean_line_confidence(data)
            
            if mlc > best_eval:
                best_eval = mlc
                best_rotation = r
            if output:
                print(' > rotation:', r , 'mlc:', round(mlc, 2))
                
        elif eval_method == 'wer':
            hypothesis = ' '.join(data[data['text'].notnull()].text.to_list())
            
            wer = word_error_rate(ground_truth, hypothesis)
            
            if wer < best_eval:
                best_eval = wer
                best_rotation = r
            if output:
                print(' > rotation:', r , 'wer:', round(wer, 2))
             
         
    time_all = round(time.time() - start, 2)
    return [best_eval, best_rotation, file_path, time_all]


# -

def get_shape_accuracy(file_path, shape):
    try:
        from PIL import Image
        img = Image.open(file_path)

        new_size = (int(img.size[0]*shape), int(img.size[1]*shape))
        img = img.resize(new_size)

        data_binary = pytesseract.image_to_data(img, output_type='data.frame', lang='eng')
        mlc = calc_mean_line_confidence(data_binary)
        
        return [file_path, shape, mlc]
    except Exception as e:
        print(e)
        return [file_path, shape, None]

# ## Helpers

# +
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''
    breaks the function func after timeout_duration seconds
    # args = non named
    # kwargs = named arguments
    '''
    import signal
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        print("Timeout after sec.", timeout_duration)
        result = default
    finally:
        signal.alarm(0)
    return result

def timeout_test(sleeptime):
    from time import sleep
    sleep(sleeptime)
    return "yeah"


def divide_chunks(l, n): 
    '''
    split list l into n batches and put the rest in n-1
    '''
    for i in range(0, len(l), n):  
        yield l[i:i + n]



# -

# # Multiprocessing rotation apply functions
# #### (in order)
# ## 2.
# - 1. optimal_image_shape_determination
# - 2. export_best_shapes
# - 3. shape_rotation_determination
# - 4. apply_rotation_correction

# +
def image_shape_determination(valid_files_list, shapes, N_CPU, batch_size, results_path):
    '''
    > 2.1
    shapes - list of percentages taken of the original image
    results_path - source
    '''
    
    # get new files
    col_names = ['file','shape','mlc']
    new_files = get_new_files_to_be_processed(path=results_path, col_names=col_names, index_col='file', all_files=valid_files_list)

    args_1 = []
    args_2 = []
    for file in new_files:
        for shape in shapes:
            args_1.append(file)
            args_2.append(shape) 

    new_files_batches = list(divide_chunks(args_1, batch_size)) 
    new_shapes_batches = list(divide_chunks(args_2, batch_size)) 
    print('total batches', len(new_files_batches), 'batch-size', batch_size)
    
    if len(new_files_batches) > 0:
        for batch in range(0,len(new_files_batches)):
            args_1 = new_files_batches[batch]
            args_2 = new_shapes_batches[batch]
            all_args = list(zip(args_1, args_2))

            print("Starting Batch..", batch)
            pool = multiprocessing.Pool()
            results = pool.starmap(get_shape_accuracy, all_args)
            cv2.destroyAllWindows()

            with open(results_path, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(results)

            pool.close()
            pool.terminate()

            
def export_best_shapes(SHAPES_PATH, BEST_SHAPES_PATH):
    ''' 
    > 2.2 
    SHAPES_PATH - source
    BEST_SHAPES_PATH - export results
    '''
    
    df_res = pd.read_csv(SHAPES_PATH)

    # fill nan mlc values
    df_res.mlc.fillna(value=0, inplace=True)

    # for plot labels
    df_res['file_ending'] = df_res.file.apply(lambda x: os.path.basename(x))
    df_res['file_ending'] = df_res['file_ending'].apply(lambda x: x.replace('TRANS',''))

    # searching for best resize within group
    df_best = df_res.groupby(['file']).apply(lambda grp: grp.nlargest(1, 'mlc'))
    df_best = df_best.reset_index(drop=True) # reset row index after groupby
    
    df_best.to_csv(BEST_SHAPES_PATH, index=False)
    
    
def shape_rotation_determination(BEST_SHAPES_PATH, SHAPE_ROTATION_RESULTS_PATH, N_CPU, batch_size):
    ''' 
    > 2.3 
    BEST_SHAPES_PATH - source
    SHAPE_ROTATION_RESULTS_PATH - export results
    '''
    
    # load source data
    df_best = pd.read_csv(BEST_SHAPES_PATH)
    
    # all filenames
    valid_files_list = df_best.file.to_list()
    
    # get new files
    col_names = ['mlc','Rotate','file', 'time']
    new_files = get_new_files_to_be_processed(path=SHAPE_ROTATION_RESULTS_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)

    # file
    new_files_batches = list(divide_chunks(new_files, batch_size)) 

    # shape
    all_shapes = df_best[df_best['file'].isin(new_files)]['shape'].to_list()
    new_shapes_batches = list(divide_chunks(all_shapes, batch_size)) 

    print('total batches', len(new_files_batches), 'batch-size', batch_size)
    
    if len(new_files_batches) > 0:
        for batch in range(0,len(new_files_batches)):
            args_1 = new_files_batches[batch]
            args_2 = new_shapes_batches[batch]

            all_args = list(zip(args_1, args_2))
            #multiprocessing.set_start_method('spawn')

            print("Starting Batch..", batch)
            pool = multiprocessing.Pool()
            results = pool.starmap(get_best_rotation, all_args)
            cv2.destroyAllWindows()

            with open(SHAPE_ROTATION_RESULTS_PATH, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(results)

            pool.close()
            pool.terminate()
            
            
def apply_rotation_correction(SHAPE_ROTATION_RESULTS_PATH, APPLY_SHAPE_ROTATION_RESULTS_PATH):
    '''
    > 2.4
    SHAPE_ROTATION_RESULTS_PATH - source
    APPLY_SHAPE_ROTATION_RESULTS_PATH - export
    '''

    import parmap

    ## read correction results
    df_corrections = pd.read_csv(SHAPE_ROTATION_RESULTS_PATH)
    #display(df_corrections.head())
    #display(df_corrections.Rotate.value_counts())

    # apply only once
    if not os.path.exists(APPLY_SHAPE_ROTATION_RESULTS_PATH):
        print(APPLY_SHAPE_ROTATION_RESULTS_PATH, ' does not exist -> apply rotation on all images')
        '''
        apply rotation to ALL files
        tskes 2 min (threadded with setting below)
        '''
        N_CPU = multiprocessing.cpu_count() - 10

        args_1 = df_corrections.file.to_list()
        args_2 = df_corrections.Rotate.to_list()
        all_args = list(zip(args_1, args_2))

        results = parmap.starmap(correct_img_rotation, all_args, 
                              pm_pbar=True,
                              pm_parallel=True,
                              pm_processes=N_CPU)

        pd.DataFrame(list(zip(args_1, args_2, results)), columns=['file', 'was_rotated', 'outcome']).to_csv(APPLY_SHAPE_ROTATION_RESULTS_PATH, index=False)
        
    else:
        print('already applied rotation correction to data!')
# -


