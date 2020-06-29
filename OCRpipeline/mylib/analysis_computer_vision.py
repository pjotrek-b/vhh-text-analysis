# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import csv
import PyPDF2

# +
'''replikas'''
def open_pil_image(img_path):
    try:
        from PIL import Image
        img = Image.open(img_path)
        return img
    except Exception as e:
        print(e, 'could not load image from desired path')
        
def save_pil_image(pil_img, to_path):
    try:
        from PIL import Image
        pil_img.save(to_path)
    except Exception as e:
        print(e, 'could not save image to desired path')

def save_bytes_image(bytes_img, to_path):
    try:
        from PIL import Image
        img = Image.fromarray(bytes_img)
        img.save(to_path)
    except Exception as e:
        print(e, 'could not save image to desired path')
    
def delete_image(img_path):
    try:
        import os
        os.remove(img_path)
    except Exception as e:
        print(e, 'wrong path?')

def divide_chunks(l, n): 
    '''
    split list l into n batches and put the rest in n-1
    '''
    for i in range(0, len(l), n):  
        yield l[i:i + n]
        
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


# -

# ### Tesseract 
# quick helpers

def tess_img_to_string(img_path, batch=False):
    '''
    from path to string
    '''
    from PIL import Image
    img = Image.open(img_path)
    out = pytesseract.image_to_string(img, lang='eng')
    return out


# ## Computer Vision CV
# color preprocessing
#
# - crop pil image
# - image to grayscale
# - image to binary
#     - notmal threshold
#     - adaptive
# - mean line confidence
# - image stats
# - grid search
#     - to find optimal binarization parameters

# +
def center_crop_pil(pil_img, crop_format=(500,500)):
    '''
    center crop a PIL image to the desired crop_format (tuple)
    '''
    w,h = pil_img.size 
    new_w = crop_format[0]
    new_h = crop_format[1]

    left = (w-new_w)/2
    top = (h-new_h)/2
    right = (w+new_w)/2
    bottom = (h+new_h)/2
    
    center_cropped = pil_img.crop((left, top, right, bottom))
    
    return center_cropped

def img_to_grayscale(img_path, to_format):
    '''
    transform image to grayscale
    LA removes the alpha channel and converts to grayscale
    '''
    output_filename = os.path.split(img_path)[0]+'/'+os.path.splitext(os.path.basename(img_path))[0]+'.'+to_format
    try:
        from PIL import Image
        options = 'L'
        img = Image.open(img_path).convert(options)
        img.save(output_filename) 
        return [img_path, True]
    except:
        print("Error: Cannot transform to grayscale..")
        return [img_path, False]
        
def grayscale_to_binary(gray_img_path, thresh=127):
    '''
    load as grayscale
    apply binarization with treshold
    return b&w image
    '''
    gray_img = cv2.imread(gray_img_path, 0)
    (thresh, binary_img) = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    return binary_img

def adaptive_binary(img_path, size, c, adaptiveMethod=0):
    '''
    params for adaptive binarization
    size = Size of a pixel neighborhood that is used to calculate a threshold value for the pixel
    c = Constant substracted from the mean or weighted mean
    adaptiveMethod (ADAPTIVE_THRESH_GAUSSIAN_C or ADAPTIVE_THRESH_MEAN_C)
    return binarized image
    '''
    try:
        gray_img = cv2.imread(img_path, 0)
        if adaptiveMethod == 0:
            img_binary = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY, size, c)
            return img_binary
        elif adaptiveMethod == 1:
            img_binary = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY, size, c)
            return img_binary
    except:
        print("Error: adaptive_binary: wrong parameters..")
        return None


# -

# ### Mean Line Confidence

# +
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

def image_to_data_stats(df_data, output=False):
    '''
    df_data = output from tesseract function image_to_data
    return result stats from image_to_data
    '''
    length = len(list(df_data[df_data.text.notnull()].text))
    text = list(df_data[df_data.text.notnull()].text) 
    mlc = calc_mean_line_confidence(df_data)
    
    if output:
        print('\n Result length:', length,'\n', 'Mean Line Confidence (mlc)', mlc, '\n\n', text)
    return length, text, mlc


# -

# ### Grid Search

# +
# rewrite gs for wer 
def grid_search_adaptive_binarization(img_path, cv_adaptive_sizes=[15,25], cv_adaptive_cs=[10], cv_adaptive_methods=['ADAPTIVE_THRESH_MEAN_C'], output=True, eval_method='mlc', ground_truth_path=None):
    '''
    try each size and C value to find best adaptive binarization params
    eval_method = [mlc, wer]
    evaluation measurement = WORD ERROR RATE
    return dict of results
    '''
    
    if eval_method not in ['mlc', 'wer']:
        print('error: invalid eval_method')
        return False
    
    if eval_method == 'wer' and ground_truth_path is not None:
        # load
        try:
            ground_truth = open(ground_truth_path, 'r').read()
        except:
            print('error: cannot load ground truth from .txt')
            
    if eval_method == 'wer' and ground_truth_path is None:
        print('error: please provide a gound truth text if evaluation method == wer')
        return False
    
    import time
    import numpy as np
    print_time_once = True
    gridsearch_result = []
    
    for method in cv_adaptive_methods:
        for c in cv_adaptive_cs:
            for size in cv_adaptive_sizes:
                start_time = time.time()
                
                # apply treshold
                img_bin_adapt = adaptive_binary(img_path, size, c, method)
                
                # apply ocr
                df_data = pytesseract.image_to_data(img_bin_adapt, output_type='data.frame')

                # measurement
                if eval_method == 'mlc':
                    try:
                        length, text, measure = image_to_data_stats(df_data, output=False)
                    except(e):
                        print('error: mean line confidence', e)
                        measure = np.nan
                        length = np.nan
                        
                elif eval_method == 'wer':
                    try: 
                        hypothesis = ' '.join(df_data[df_data['text'].notnull()].text.to_list())
                        length, _, _ = image_to_data_stats(df_data, output=False)
                        measure = word_error_rate(ground_truth, hypothesis)
                    except:
                        measure = np.nan
                        length = np.nan
             
                # add gs data
                gridsearch_result.append({'length':length, eval_method:measure, 'size':size, 'c':c, 'method':method})
                
                # make the time stop iterat
                time_end = round((time.time() - start_time),2)
                
                if print_time_once:
                    cv_iterations = len(cv_adaptive_sizes)*len(cv_adaptive_cs)*len(cv_adaptive_methods)
                    print('eval_method:', eval_method, ' nr. of iterations:', cv_iterations, 'est. time (min):', round(cv_iterations*time_end/60, 2))
                    print_time_once = False
                
                if output:
                    print('GS>','method', method, 'size', size, 'c', c, eval_method, measure, '\t time:', time_end)
                    plot_img(img_bin_adapt, 150) 
                    plt.pause(0.05)
    
    '''return dict and best index with best params'''
    if eval_method == 'mlc':
        best_idx, best_params = max(enumerate(gridsearch_result), key=lambda item: item[1][eval_method])
    
    elif eval_method == 'wer':
        best_idx, best_params = min(enumerate(gridsearch_result), key=lambda item: item[1][eval_method])
    
    return gridsearch_result, best_idx, best_params



def collect_gs_data(file_path, cv_adaptive_sizes, cv_adaptive_cs, cv_adaptive_methods, eval_method='mlc', ground_truth_path=None):
    '''
    gridsearch form data of func: grid_search_adaptive_binarization 
        for adaptive binarization
    file_path - path to file
    cv_adaptive_sizes - GS parameter 
    cv_adaptive_cs - GS parameter
    cv_adaptive_methods - GS parameter
    return row with results + benchmark 
        for data gathering 
        for best adaptive binarization settings
    '''
    try:
        # benchmark data
        benchmark_df_data = pytesseract.image_to_data(Image.open(file_path), output_type='data.frame')
        
        if eval_method == 'mlc':
            benchmark_length, benchmark_text, benchmark_measure = image_to_data_stats(benchmark_df_data, output=False)
            
        elif eval_method == 'wer':
            try:
                ground_truth = open(ground_truth_path, 'r').read()
            except:
                print('error: cannot load ground truth from .txt')
            
            hypothesis = ' '.join(benchmark_df_data[benchmark_df_data['text'].notnull()].text.to_list())
            benchmark_length, _, _ = image_to_data_stats(benchmark_df_data, output=False)
            benchmark_measure = word_error_rate(ground_truth, hypothesis)

        # gs and retrieve best result
        try:
            gs_result, best_idx, best_params = grid_search_adaptive_binarization(file_path, cv_adaptive_sizes, cv_adaptive_cs, cv_adaptive_methods, \
                                                       output=False, eval_method=eval_method, ground_truth_path=ground_truth_path)
            
        except Exception as e:
            print("error: gridsearch parameters", e)
            
        result_row = [file_path]
        result_row.extend(list(best_params.values()))
        result_row.extend([benchmark_length, benchmark_measure])
        return result_row
    
    except Exception as e:
        result_row = [file_path]
        result_row.extend([np.nan]*7)
        print('err: collect gs-data', e)
        return result_row


# -
def get_best_gs(gs_result_dict):
    '''
    param = return-data[0] from grid_search_adaptive_binarization function
    new metric: length x mlc as best index
    '''
    tmp = pd.DataFrame(gs_result_dict)
    tmp['mlcXlen'] = tmp.mlc * tmp.length
    max_index = tmp['mlcXlen'].idxmax()
    best_row = tmp[tmp.index == max_index]
    return best_row


# ### GS analysis Plot before-after

def compare_befor_after_binarization(row, crop_format = (900,900)):
    '''
    take row from gridsearch results 
    print stats and image
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    tmp_f_name_og = 'temp_images/tmp_original.jpg'
    tmp_f_name_bin = 'temp_images/tmp_binarized.jpg'
    
    '''benchmark'''
    data_raw = pytesseract.image_to_data(row.file, output_type='data.frame')    
    r_length, r_text, r_mean_confidence = image_to_data_stats(data_raw, output=False)

    original = open_pil_image(row.file)
    save_pil_image(original, tmp_f_name_og)
    
    '''binarized'''
    binary = adaptive_binary(row.file, row['size'], row['c'], row['method'])
    binary = Image.fromarray(binary)
    save_pil_image(binary, tmp_f_name_bin)
    
    data_binary = pytesseract.image_to_data(tmp_f_name_bin, output_type='data.frame')
    b_length, b_text, b_mean_confidence = image_to_data_stats(data_binary, output=False)

    '''plot'''
    comparison_df = pd.DataFrame([[os.path.basename(row.file), 'benchmark',r_mean_confidence,r_length], \
                                  [os.path.basename(row.file), 'binarized',b_mean_confidence,b_length]], \
                                 columns=['file','img','mlc','length'])
    
    original_crop = center_crop_pil(original, crop_format)
    binary_crop = center_crop_pil(binary, crop_format)

    fig, axs = plt.subplots(1,2, figsize=(15,5), facecolor=None, edgecolor='yellow')
    axs[0].imshow(original_crop, cmap='Greys_r')
    axs[0].set_title("original")
    axs[1].imshow(binary_crop, cmap='Greys_r')
    axs[1].set_title("binarized")
    fig.suptitle('Before - After Binarization', fontsize=16)
    
    display(comparison_df)

# ### 4. final extraction pipeline
# - extract and add symspell suggestions and wordsegmentation columns


# +
def get_spelling_correction(word, n_best, sym_spell):
    try:
        if (type(word) == str) and word != '':
            from symspellpy import SymSpell, Verbosity
            # spelling suggestions
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            suggestions = suggestions[:n_best]
            all_suggestions = [s.term for s in suggestions]
            if len(all_suggestions) == 0:
                return np.nan
            return all_suggestions
        else:
            return False
    except Exception as e:
        return False
    
def get_word_segmentation(word, sym_spell):
    try:
        if (type(word) == str) and word != '':
            # word segmentation 
            all_segmentation = sym_spell.word_segmentation(word)
            all_segmentation = all_segmentation.corrected_string.split(' ')
            return all_segmentation
        else:
            return False
    except Exception as e:
        return False

def extract_text_from_image(img_row, sym_spell, config='', safe_temp=None, add_filename=True, plot=False, rotate=0, use_binarize=True):
    # yes binarize
    if use_binarize:
        img_binary = adaptive_binary(img_row[0], img_row[3], img_row[4], img_row[5])
        img_binary = Image.fromarray(img_binary)
    else:
        # no binarize
        img_binary = cv2.imread(img_row, 0)
        img_binary = Image.fromarray(img_binary)
    
    # rotate
    img_binary = img_binary.rotate(rotate, expand=True)
        
    # safe bin image
    if safe_temp != None:
        save_pil_image(img_binary, safe_temp)
    
    if plot:
        plot_img(img_binary)
        
    df_data = pytesseract.image_to_data(img_binary, config=config, lang='eng+deu+fra', output_type='data.frame')

    # lowcase for spelling correction
    df_data['text_low'] = df_data['text'].astype(str).str.lower()
    
    # sc - spelling correction 
    df_data['symspell_sc'] = df_data.apply(lambda row: get_spelling_correction(row.text_low, 4, sym_spell), axis=1)

    # ws - word segmentation
    try:
        df_data['symspell_ws'] = df_data[df_data['symspell_sc'].isnull()].apply(lambda row: get_word_segmentation(row.text_low, sym_spell), axis=1)
    except Exception as e:
        df_data['symspell_ws'] = False
    
    # add filename to data
    if add_filename:
        if use_binarize:
            df_data['file'] = img_row[0]    
        else:
            df_data['file'] = img_row
        
    return df_data
# -

def show_human_viewable(file_path):
    '''
    return corrected PIL img and rotation angle
    '''
    im = Image.open(file_path)
    width, height = im.size
    if width > height:
        #print('flipped 90°')
        im = im.rotate(90, expand=1)
        rotate = 90 
    else:
        #print('straight 0°')
        rotate = 0
    return im, rotate

# -

def word_error_rate(ground_truth, hypothesis):
    from jiwer import wer
    error = wer(ground_truth, hypothesis)
    return error


# # Multiprocessing Computer Vision Task
# #### 1. grayscaling
# #### 3. binarization gridsearch
# #### 4. export
# #### 5. ocr showcase extract

# +
import multiprocessing
from multiprocessing import Pool

'''1. grayscaling'''
def grayscale_valid_files(valid_files_list, N_CPU, GRAYSCALE_RESULTS_PATH):

    # get new files
    col_names = ['file', 'gray_status']
    new_files = get_new_files_to_be_processed(path=GRAYSCALE_RESULTS_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)
    
    
    if len(new_files) > 0:
        args_1 = new_files
        args_2 = ['jpg']*len(args_1)
        all_args = zip(args_1, args_2)

        print('start grayscaling..')
        print('grayscaling files: ', len(args_1), ' cpus', N_CPU)
        pool = Pool(processes=N_CPU)

        res_gray = pool.starmap_async(img_to_grayscale, all_args)
        results_gray = res_gray.get()
        print('..done grayscaling')
        
        
        with open(GRAYSCALE_RESULTS_PATH, 'a', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(results_gray)

''' 3.0 filter before gridsearch '''        
def pipeline_filter_quality(SHAPE_ROTATION_RESULTS_PATH, MLC_THRESHOLD): 
    # read correction results
    df = pd.read_csv(SHAPE_ROTATION_RESULTS_PATH, header=0)
    
    # filter low mlc's files
    df_below = df[df.mlc < MLC_THRESHOLD]
    
    # valid files
    df_valid_files = df.drop(df_below.index)
    filtered_valid_files_list = list(df_valid_files.file)
    print('filter valid files:\t', df_valid_files.shape[0], 
          '\nfilter below threshold:', df_below.shape[0], '(irrelevant)')

    return filtered_valid_files_list

''' 3. binarization gridsearch '''
def pipeline_binarization_gridsearch(valid_files_list, measure_method, cv_adaptive_sizes, cv_adaptive_cs, cv_adaptive_methods, GS_PATH, N_CPU, batch_size):
    print('3.pipeline_binarization_gridsearch')
    
    # get new files
    col_names = ['file', 'length', 'measure', 'size', 'c', 'method', 'benchmark_length', 'benchmark_measure']
    new_files = get_new_files_to_be_processed(path=GS_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)

    if measure_method is 'wer':
        # get files and transkript
        img_paths = []
        groundtruth_paths = []
        for i in new_files:    
            tmp_groundtruth = os.path.splitext(i)[0]+'_transkript.txt'
            if os.path.exists(tmp_groundtruth):
                groundtruth_paths.append(tmp_groundtruth)
                img_paths.append(i)

    elif measure_method is 'mlc': 
        # get files and set transkript to none
        img_paths = new_files
        groundtruth_paths = len(img_paths)*[None]

    new_files_batches = list(divide_chunks(img_paths, batch_size)) 
    new_transkripts_batches = list(divide_chunks(groundtruth_paths, batch_size)) 
    print('total batches:', len(new_files_batches), '\t measure_method:', measure_method)
     
    cv2.destroyAllWindows()
    if len(new_files_batches) > 0:
        for batch in range(0, len(new_files_batches)):
            args_1 = new_files_batches[batch]
            args_2 = len(args_1)*[cv_adaptive_sizes] 
            args_3 = len(args_1)*[cv_adaptive_cs] 
            args_4 = len(args_1)*[cv_adaptive_methods] 
            args_5 = len(args_1)*[measure_method]
            args_6 = new_transkripts_batches[batch] 
            all_args = list(zip(args_1, args_2, args_3, args_4, args_5, args_6))

            print("Starting Batch..", batch)
            pool = multiprocessing.Pool()
            results = pool.starmap(collect_gs_data, all_args)

            with open(GS_PATH, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(results)

            cv2.destroyAllWindows()
            pool.close()
            pool.terminate()

            
            
''' 4. export results'''            
def pipeline_ocr_export(GS_PATH, EXPORT_RESULTS_PATH, tess_config, batch_size, limit=None, use_binarize=True):
    '''
    run this function binarization is needed
        GS_PATH - source (to binarization parameters)
        EXPORT_RESULTS_PATH - export
    '''
    
    # load and init symspell library
    import pkg_resources
    from symspellpy import SymSpell, Verbosity
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # load gridsearch data
    df_valid_files = pd.read_csv(GS_PATH)
    valid_files_list = list(df_valid_files.file)
    print('total files:\t', len(valid_files_list))

    # get new files
    col_names = ['row_nr', 'level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
           'left', 'top', 'width', 'height', 'conf', 'text', 'text_low', 'symspell_sc',
           'symspell_ws', 'file']
    new_files = get_new_files_to_be_processed(path=EXPORT_RESULTS_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)

    # get array shaped input parameters
    new_files = df_valid_files[df_valid_files.file.isin(new_files)].values
    if limit != None:
        new_files = new_files[:limit]
        

    new_files_batches = list(divide_chunks(new_files, batch_size)) 
    print('total batches', len(new_files_batches), 'total files', len(new_files))
    
    conf = tess_config
    if len(new_files_batches) > 0:
        for batch in new_files_batches[:]:
            args_1 = batch
            args_2 = len(args_1)*[sym_spell]
            args_3 = len(args_1)*[conf]
            args_4 = len(args_1)*[None]
            args_5 = len(args_1)*[True]
            args_6 = len(args_1)*[False]
            args_7 = len(args_1)*[0]
            args_8 = len(args_1)*[use_binarize]
            all_args = list(zip(args_1, args_2, args_3, args_4, args_5, args_6, args_7, args_8))

            print("Starting Batch..")
            pool = multiprocessing.Pool()
            results = pool.starmap(extract_text_from_image, all_args)
            
            df_tmp_results = pd.concat(results).to_records()

            with open(EXPORT_RESULTS_PATH, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(df_tmp_results)

            cv2.destroyAllWindows()
            pool.close()
            pool.terminate()


''' 4.1 export results without binarization'''            
def pipeline_ocr_export_no_binarization(SHAPE_ROTATION_RESULTS_PATH, EXPORT_RESULTS_PATH, tess_config, batch_size, limit=None):
    '''
    run this function if binarization has already been applied or is not needed
        SHAPE_ROTATION_RESULTS_PATH - source (files from rotation output)
        EXPORT_RESULTS_PATH - export
    '''
    
    # load and init symspell library
    import pkg_resources
    from symspellpy import SymSpell, Verbosity
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # load gridsearch data
    df_valid_files = pd.read_csv(SHAPE_ROTATION_RESULTS_PATH)
    valid_files_list = list(df_valid_files.file)
    print('total files:\t', len(valid_files_list))

    # get new files
    col_names = ['row_nr', 'level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
           'left', 'top', 'width', 'height', 'conf', 'text', 'text_low', 'symspell_sc',
           'symspell_ws', 'file']
    new_files = get_new_files_to_be_processed(path=EXPORT_RESULTS_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)

    # get list shaped input only filename
    new_files = df_valid_files[df_valid_files.file.isin(new_files)].file.to_list()
    if limit != None:
        new_files = new_files[:limit]
                
    
    new_files_batches = list(divide_chunks(new_files, batch_size)) 
    print('total batches', len(new_files_batches), 'total files', len(new_files))
    
    conf = tess_config
    if len(new_files_batches) > 0:
        for batch in new_files_batches[:]:
            args_1 = batch
            args_2 = len(args_1)*[sym_spell]
            args_3 = len(args_1)*[conf]
            args_4 = len(args_1)*[None]
            args_5 = len(args_1)*[True]
            args_6 = len(args_1)*[False]
            args_7 = len(args_1)*[0]
            args_8 = len(args_1)*[False]
            all_args = list(zip(args_1, args_2, args_3, args_4, args_5, args_6, args_7, args_8))

            print("Starting Batch..")
            pool = multiprocessing.Pool()
            results = pool.starmap(extract_text_from_image, all_args)

            df_tmp_results = pd.concat(results).to_records()

            with open(EXPORT_RESULTS_PATH, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(df_tmp_results)

            cv2.destroyAllWindows()
            pool.close()
            pool.terminate()


def extract_pdf_content(file_path):
    '''
    open and read pdf
    return list of pages of string content
    '''
    import PyPDF2
    content = []
    result_row = [file_path]
    
    try:
        pdf_file = open(file_path, 'rb')
        read_pdf = PyPDF2.PdfFileReader(pdf_file)
        pages = read_pdf.getNumPages()

        for i in range(pages):
            page = read_pdf.getPage(i)
            page_content = page.extractText()
            content.append(page_content)
            
        result_row.append(content)
        return result_row
    
    except Exception as e:
        print("error: gridsearch parameters", e)
        result_row.append([False])
        return result_row
    
    
''' 4.2 export pdf content (already ocr'd) '''
def pipeline_extract_pdf_content(valid_files_list, EXPORT_RESULTS_PATH, batch_size, limit=None):
    '''
    function extract_pdf_content
    '''
    print('total files:\t', len(valid_files_list))
    
    # get new files
    col_names = ['file', 'data']
    new_files = get_new_files_to_be_processed(path=EXPORT_RESULTS_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)
    
    # limit
    if limit != None:
        new_files = new_files[:limit]
        
    new_files_batches = list(divide_chunks(new_files, batch_size)) 
    print('total batches', len(new_files_batches), 'total files', len(new_files))
    
    if len(new_files_batches) > 0:
        for batch in new_files_batches[:]:
            args_1 = batch
            all_args = list(zip(args_1))

            print("Starting Batch..")
            pool = multiprocessing.Pool()
            results = pool.starmap(extract_pdf_content, all_args)

            with open(EXPORT_RESULTS_PATH, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(results)

            pool.close()
            pool.terminate()
        print("done")

def single_hocr_extract(row, HOCR_DIR, sym_spell, conf):
    straight_img, straight_angle = show_human_viewable(row[0])
    # paths
    filename = row[0].lstrip('.')
    to_data_csv = HOCR_DIR+os.path.splitext(filename)[0].replace('/','_').replace(' ','')+'.csv'
    to_img_path_bin = HOCR_DIR+os.path.splitext(filename)[0].replace('/','_').replace(' ','')+'_bin.jpg'
    to_img_path_og = HOCR_DIR+os.path.splitext(filename)[0].replace('/','_').replace(' ','')+'.jpg'
    
    # save binarized and create data
    res_df_data = extract_text_from_image(row, sym_spell, conf, \
                                          safe_temp=to_img_path_bin, \
                                          add_filename=False, \
                                          plot=False, \
                                          rotate=straight_angle, \
                                          use_binarize=True)
    # save csv and original
    res_df_data.to_csv(to_data_csv)
    save_pil_image(straight_img, to_img_path_og)
    return row[0], to_data_csv


''' 5. showcaser data extract '''
def extract_for_hocs_showcase(GS_PATH, HOCR_RESULTS_PATH, HOCR_DIR, conf, limit, N_CPU, batch_size):
    '''
    df_gs - input dataframe from gridsearch
    '''
    # init symspell library
    import pkg_resources
    from symspellpy import SymSpell, Verbosity
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    
    # index list of filenames
    filename_path = HOCR_DIR+'filenames_index.csv'
    files = ['choose file>']
    language = 'eng+deu+fra'
    
    # filter
    df_valid_files = pd.read_csv(GS_PATH)
    df_valid_files = df_valid_files[df_valid_files.length > 100]
    df_valid_files = df_valid_files.sort_values(by=["measure"], ascending=False)
    
    valid_files_list = list(df_valid_files.file)
    print('total files:\t', len(valid_files_list))

    # get new files
    col_names = ['file','name']
    new_files = get_new_files_to_be_processed(path=HOCR_RESULTS_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)

    # get array shaped input parameters
    new_files = df_valid_files[df_valid_files.file.isin(new_files)].values
    if limit != None:
        new_files = new_files[:limit]
 
    new_files_batches = list(divide_chunks(new_files, batch_size)) 
    print('total batches', len(new_files_batches), 'total files', len(new_files))
    
    files = []
    cv2.destroyAllWindows()
    if len(new_files_batches) > 0:
        for batch in new_files_batches[:]:
            args_1 = batch
            args_2 = len(args_1)*[HOCR_DIR]
            args_3 = len(args_1)*[sym_spell]
            args_4 = len(args_1)*[conf]
            all_args = list(zip(args_1, args_2, args_3, args_4))
            
            print("Starting Batch..")
            pool = multiprocessing.Pool(processes=N_CPU)
            results = pool.starmap(single_hocr_extract, all_args)

            files.extend(results)
            with open(HOCR_RESULTS_PATH, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(results)

            cv2.destroyAllWindows()
            pool.close()
            pool.terminate()
 
        filename_path = HOCR_DIR+'filenames_index.csv'
        files = ['data/'+i[0] for i in files]
        pd.DataFrame({'filename':files}).to_csv(filename_path, index=False) 
# -

# - 6

def get_first_index(row):
    '''
    get first index of list in pandas column
    '''
    from ast import literal_eval
    try:
        return literal_eval(row)[0]
    except Exception as e:
        return ''
    return row

'''
6. extract to single .json files from big .csv
'''
def results_json_split(EXPORT_RESULTS_PATH, JSON_DIR):
    '''
    - export to VHH standard specs
    - split .csv of multiple ocr file results
    - into multiple json files with two columns
    '''
    import json
    df_export = pd.read_csv(EXPORT_RESULTS_PATH)
    dfs = dict(tuple(df_export.groupby('file')))
    print('splitting', len(dfs), 'dataframes')
    relevant_columns = ['text','symspell_sc']
    
    for i, (filename, value) in enumerate(dfs.items()):
        #print(">> ",i, dfs[filename].shape, filename)
        try:
            json_path = JSON_DIR+os.path.splitext(filename)[0].replace('/','_').replace(' ','')+'.json'
            tmp_df = dfs[filename][relevant_columns] 
            tmp_df.columns = ['ocrText','ocrTextWithCorrections']
            tmp_df = tmp_df[tmp_df['ocrText'].notnull()]
            tmp_df['ocrTextWithCorrections'] = tmp_df['ocrTextWithCorrections'].apply(lambda row: get_first_index(row))
            
            data = ({'ocrText': ' '.join(tmp_df['ocrText']),
                     'ocrTextWithCorrections': ' '.join(tmp_df['ocrTextWithCorrections'])
                    })
            with open(json_path, 'w') as fp:
                json.dump(data, fp)
            
        except Exception as e:
            print("ERROR__1: json:", e)

# - 7

'''
7. extract single file to .pdf
'''
def image_to_pdf(img_row, PDF_DIR, config=''):
    '''
    binarize and export to annotated pdf
    '''
    try:
        img_path = img_row[0]
        img_binary = adaptive_binary(img_path, img_row[3], img_row[4], img_row[5])
        img_binary = Image.fromarray(img_binary)

        pdf = pytesseract.image_to_pdf_or_hocr(img_binary, lang='eng+deu+fra', config=config, extension='pdf')

        pdf_path = PDF_DIR+os.path.splitext(img_path)[0].replace('/','_').replace(' ','')+'.pdf'
        with open(pdf_path, 'w+b') as f:
            f.write(pdf) 
        return [img_path, True]
    except:
        return [img_path, False]
        

'''
7. extract all to single .pdf files for each file
'''
def pipeline_export_all_to_pdf(GS_PATH, EXPORT_PDF_EXPORT_PATH, PDF_DIR, tess_config, batch_size):    
    # load gridsearch data
    df_valid_files = pd.read_csv(GS_PATH)
    valid_files_list = list(df_valid_files.file)
    print('total files:\t', len(valid_files_list))

    # get new files
    col_names = ['file', 'success']
    new_files = get_new_files_to_be_processed(path=EXPORT_PDF_EXPORT_PATH, col_names=col_names, index_col='file', all_files=valid_files_list)

    # get array shaped input parameters
    new_files = df_valid_files[df_valid_files.file.isin(new_files)].values
        
    new_files_batches = list(divide_chunks(new_files, batch_size)) 
    print('total batches', len(new_files_batches), 'total files', len(new_files))
     
    conf = tess_config
    
    if len(new_files_batches) > 0:
        for batch in new_files_batches[:]:
            args_1 = batch
            args_2 = len(args_1)*[PDF_DIR] 
            args_3 = len(args_1)*[tess_config] 
            all_args = list(zip(args_1, args_2, args_3))

            print("Starting Batch..")
            pool = multiprocessing.Pool()
            results = pool.starmap(image_to_pdf, all_args)

            with open(EXPORT_PDF_EXPORT_PATH, 'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerows(results)

            pool.close()
            pool.terminate()
        print("done")

# -
