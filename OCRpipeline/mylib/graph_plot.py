# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ### Basic Image Plot

# +
def plot_img_path(img_path, dpi=80):
    '''
    plot straight from image path
    '''
    from PIL import Image
    img = Image.open(img_path, 'r')
    plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='y', edgecolor='k')
    plt.imshow(img, cmap='Greys_r')
    
    
def plot_img(img, dpi=80):
    '''
    img - already loaded from PIL
    '''
    plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='y', edgecolor='k')
    plt.imshow(img, cmap='Greys_r')
    
    
def plot_img_path_multiple(path_list, dpi=80):
    '''
    plot multiple straight from image path
    '''
    for f in path_list:
        print(f)
        plot_img_path(f, dpi)
        plt.pause(.001)   


# -

# ## Plots 
# - OSD Boundary Plot
# - single gridsearch result
# - before after gridsearch

# +
import pytesseract
from pytesseract import Output
import cv2

def draw_text_boundaries(img_path, output=True):
    img = cv2.imread(img_path)

    height, width, channels = img.shape 
    x_min = width
    y_min = height
    x_max = 0
    y_max = 0

    box_data = pytesseract.image_to_data(img, output_type=Output.DATAFRAME) 
    box_data = box_data[(box_data.text.notnull()) &  (box_data.text.str.strip() != '')]
    n_boxes = box_data.index.to_list()

    for box in n_boxes:
        (x, y, w, h) = (box_data['left'][box], box_data['top'][box], box_data['width'][box], box_data['height'][box])

        if x < x_min:
            x_min = x

        if y < y_min:
            y_min = y

        if x > 1 and x+w > x_max:
            x_max = x+w

        if y > 1 and y+h > y_max:
            y_max = y+h

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)

    w_boundary = x_max - x_min
    h_boundary = y_max - y_min
    cv2.rectangle(img, (x_max, y_max), (x_max+100, y_max+100), (0, 255, 0), 20)
    cv2.rectangle(img, (x_min, y_min), (x_min+w_boundary, y_min+h_boundary), (0, 255, 0), 30)
    
    if output:
        print('Text Boundary\n')    
        print('original width\t',width)
        print('original height\t',height)
        print('x_min',x_min, 'x_max',x_max)
        print('y_min',y_min, 'y_max',y_max)
        print('w',w_boundary,'h',h_boundary)
    
    return img, box_data


# -
def plot_gridsearch_adaptive_binarization(gs_result, benchmark_mlc, best_mlc):
    '''
    gs_result param from 'grid_search_adaptive_binarization' function
    '''
    
    df = pd.DataFrame(gs_result)
    df['merged'] = 's_'+df['size'].map(str)+'_c_'+df['c'].map(str)+'_meth_'+df['method'].map(str)
    df_higher = df[df.mlc >= benchmark_mlc] 
    df_lower = df[df.mlc < benchmark_mlc] 
    df_best = df[df.mlc == best_mlc] 

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.set_ylim(df.mlc.min()-5,df.mlc.max()+5)
    plt.xticks(rotation=90)

    ax.set_xlabel('Adaptive Binarization Parameters')
    ax.set_ylabel('Mean Line Confidence (mlc)')
    ax.set_title('Gridsearch Results')


    ax.bar(df_higher.merged, df_higher.mlc, align='center', alpha=0.5, color = 'green', label='> benchmark')
    ax.bar(df_lower.merged, df_lower.mlc, align='center', alpha=0.5, color = 'blue', label='< benchmark')
    ax.bar(df_best.merged, df_best.mlc, align='center', alpha=0.8, color = 'darkgreen', label='best gs')
    
    if df_best.shape[0] != 0 and df.shape[1] != 0:
        ax.legend(framealpha=0.5)

    plt.show()
