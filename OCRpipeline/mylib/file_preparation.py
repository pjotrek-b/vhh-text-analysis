import shutil
import os
import io
import pandas as pd


def cool_func():
    print("quite cool yeah?")


# ### PIL Image

# +

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


# -

# ### File Transformations

def get_filename(path, with_extension=False):
    if with_extension:
        return os.path.basename(path)
    else:
        return os.path.splitext(os.path.basename(path))[0]


# +
#0
def create_data_work_directory(og, work, overwrite=False):
    '''
    create working data directory 
    to preserve the og data
    '''
    if not os.path.exists(work):
        shutil.copytree(og, work)
    else:
        if overwrite:
            shutil.rmtree(work)
            shutil.copytree(og, work)
            print("overwriting existing workdir..")
        else:
            print('workdir already exists..')
            
# 1 - new
def get_valid_files(DIR, filter_extensions=None, filter_trash=False):
    '''
    faster version
    return valid files of dir
    filter - file extensions
    filter - trash folder
    '''
    # get directory file-structure 
    file_list_data = []
    for root, dir_name, files in os.walk(DIR):
        files = [f for f in files if not f[0] == '.']
        dir_name[:] = [d for d in dir_name if not d[0] == '.']
        
        for f in files:
            filename, file_extension = os.path.splitext(f)
            file_list_data.append([os.path.join(root, f), file_extension])        

    df = pd.DataFrame(file_list_data, columns=['file','extension'])
    print('> all unique extensions:\t', df.extension.unique())

    # filter out Trash files
    if filter_trash:
        df = df[~df.file.str.contains('Trash')]

    # filter extensions
    if filter_extensions != None:
        df = df[df.extension.isin(filter_extensions)]
    print('> valid unique extensions:\t', df.extension.unique())
        
    print('> shape:\t', df.shape)
    return df     


def valid_files_df_to_dict(df_files):
    '''
    return dict without names of a get_valid_files() pandas df
    key = file, value = extension column
    '''
    keys = df_files.file.values
    values = df_files.extension.values
    files_d = dict(zip(keys, values))
    return files_d
            
# 1 - old
def list_files_in_dir(path, extensions=None):
    ''' 
    return list of all files of a directory recursivels and with a file-extension
    specify None to get all files non filtered (with any extension)
    '''
    file_dict = {}
    for obj in os.listdir(path):
        if os.path.isfile(os.path.join(path, obj)):
            if extensions is None:
                file_dict[os.path.join(path, obj)] = os.path.splitext(obj)[1][1:]
            elif os.path.splitext(obj)[1][1:] in extensions:
                file_dict[os.path.join(path, obj)] = os.path.splitext(obj)[1][1:]
        elif os.path.isdir(os.path.join(path, obj)):
            sub_file_dict = list_files_in_dir(str(os.path.join(path, obj)), extensions)
            file_dict.update(sub_file_dict)
    return file_dict



# -

# 2
def split_pdf_into_images(path, to_format='jpg', resolution=300):
    import PyPDF2 
    from wand.image import Image
    '''
    given any pdf file (multi or single page) to be split 
    into single images (png or pdf if to_pdf=True)
    saves into the same directory
    '''    
    if os.path.splitext(os.path.basename(path))[1][1:] == 'pdf':
        f_name = os.path.splitext(os.path.basename(path))[0]
        f_path = os.path.split(path)[0]+'/'
        
        if to_format == 'jpg':
            pages = Image(filename=path, resolution=300)
            try:
                for nr, page in enumerate(pages.sequence):
                    output_filename = f_path+f_name+'_'+str(nr+1)+'.jpg'

                    with Image(page) as img:
                        img.compression_quality = 99
                        img.save(filename=output_filename)
                return True
            except Exception as e:
                return False

        else:
            reader = PyPDF2.PdfFileReader(path)
            for page in range(reader.getNumPages()):
                writer = PyPDF2.PdfFileWriter()
                writer.addPage(reader.getPage(page))

                if to_format == 'pdf':
                    output_filename = f_path+f_name+'_'+str(page+1)+'.pdf'
                    with open(output_filename, 'wb') as out:
                        writer.write(out)

                elif to_format == 'png':
                    output_filename = f_path+f_name+'_'+str(page+1)+'.png'
                    pdf_bytes = io.BytesIO()
                    writer.write(pdf_bytes)
                    pdf_bytes.seek(0)

                    img = Image(file=pdf_bytes, resolution=resolution)
                    img.convert("png").save(filename = output_filename)
                    pdf_bytes.flush()


    else:
        print("file is not a .pdf", path)
        return
    return
# ### Raw 2 JPG

# +
#3
import rawpy
import imageio
import os

def raw_transform(img_path, to_format='jpg'):
    '''
    > export ARW, DNG file as jpg
    *can adapt code to export other raw extensions
    '''
    try: 
        f_name = get_filename(img_path)
        f_path = os.path.split(img_path)[0]+'/'
        output_filename = f_path+f_name+'.'+to_format

        with rawpy.imread(img_path) as raw:
            img_trans = raw.postprocess()
        imageio.imwrite(output_filename, img_trans)
        return True
    except Exception as e:
        print('error raw conversion:', e)
        return False


# -
# # pipeline file transformations
# - ARW to jpg

# +
import multiprocessing
from multiprocessing import Pool


def pipeline_split_pdf(df_files, TRANSFORM_FILE_PDF, N_CPU):
    '''
    df_files - get_valid_files pd dataframe only_pdf_files
    TRANSFORM_FILE_PDF - export 
    '''
    
    # filter ARW files
    df_only = df_files[df_files.extension == '.pdf']
    only_pdf_files = valid_files_df_to_dict(df_only) 
    print(df_only.shape[0], '.pdf files were found & need to be converted into .jpg')
        
    # multiprocessing
    if df_only.shape[0] > 0:
        args_1 = list(only_pdf_files.keys())
        args_2 = len(args_1) * ['jpg']
        args_3 = len(args_1) * [300]
        all_args = zip(args_1, args_2, args_3)
        
        print('transforming .pdf files: ', len(args_1), ' cpus', N_CPU)
        pool = Pool(processes=N_CPU)
        res = pool.starmap_async(split_pdf_into_images, all_args)
        results = res.get()
        pool.close()
        pool.terminate()
        
        # save results
        file_transform_res = list(zip(args_1, results))
        df_aws_transform_results = pd.DataFrame(file_transform_res, columns=['file','success'])
        df_aws_transform_results.to_csv(TRANSFORM_FILE_PDF)

        # select only successfully transformed and remove original ARW file
        to_remove = df_aws_transform_results[df_aws_transform_results.success == True].file.to_list() 
        failed_transformations = df_aws_transform_results[df_aws_transform_results.success == False].file.to_list() 
        for f in to_remove[:]:
            if os.path.exists(f):
                os.remove(f)   

        for f in failed_transformations[:]:
            if os.path.exists(f):
                os.remove(f) 
# -

def pipeline_transform_raw(df_files, TRANSFORM_FILE_RAW, N_CPU):
    '''
    df_files - get_valid_files pd dataframe
    TRANSFORM_FILE_RAW - export success
    '''
    # filter raw files
    raw_extensions = ['.ARW', '.DNG']
    df_only_raw = df_files[df_files.extension.isin(['.ARW', '.DNG'])]
    only_raw_files = valid_files_df_to_dict(df_only_raw) 
    print(df_only_raw.shape[0], raw_extensions, ' files were found & need to be converted into .jpg')
    

    if df_only_raw.shape[0] > 0:
        args_1 = list(only_raw_files.keys())
        all_args = zip(args_1)

        # multiprocess transform
        print('transforming .RAW files ', raw_extensions, ':', len(args_1), ' cpus', N_CPU)
        pool = Pool(processes=N_CPU)
        res = pool.starmap_async(raw_transform, all_args)
        results = res.get()
        pool.close()
        pool.terminate()
        
        # save results
        file_transform_res = list(zip(args_1, results))
        df_aws_transform_results = pd.DataFrame(file_transform_res, columns=['file','success'])
        df_aws_transform_results.to_csv(TRANSFORM_FILE_RAW)

        # select only successfully transformed and remove original .RAW file
        to_remove = df_aws_transform_results[df_aws_transform_results.success == True].file.to_list() 
        failed_transformations = df_aws_transform_results[df_aws_transform_results.success == False].file.to_list() 
        for f in to_remove[:]:
            if os.path.exists(f):
                os.remove(f)   

        for f in failed_transformations[:]:
            if os.path.exists(f):
                os.remove(f) 
# -







