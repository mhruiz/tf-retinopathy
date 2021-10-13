from lib.preprocess import process_image
from concurrent import futures
from shutil import copy2
import pandas as pd
import numpy as np
import argparse
import cv2
import os

'''
This script takes all datasets specified by arguments and creates for each of them a new directory inside 'data/processed_datasets'.
For each directory / dataset, it will create these directories:
- A new directory for all ungradable images: every ungradable image will be stored inside (although it has DR level assigned).
All images whose DR diagnosis is unknown (-1) MUST have as gradability label '0' (Ungradable)
- A new directory for each DR level present in this dataset. These directories will contain only grabable images.
- A new direcotry for all images whose FOV was not detected. These images will not be processed and they will be ignored.
There are some exceptions:
- If a dataset has no information about images gradability, 'ungradables' directory will not be created. Instead of that, an empty txt file 
named 'This dataset does not have gradability labels.txt' will be generated.
- If a dataset has all its images with their FOV detected, no directory for undetected FOV will be created.

All images that will be saved in those new directories after being processed by cropping the FOV, resizing the image to a better size (given by arguments)
(default size is 540) and padding it to form a squared shape image. The new image will have a .png file extension.

A new CSV file will be generated for each dataset, containing these columns:
    ---------------------------------------------------------------------
    image, path, DR_level, DME_level, gradability, old_size_x, old_size_y
    ---------------------------------------------------------------------

The meaning for each columns is:
- image: it's the image's file name, without file extension
- path: it's the image's full path starting from 'data/' directory. Example: 'data/original_datasets/eyepacs/10_left.jpeg'
- DR_level: it's the diagnosed DR level or grade for current image, according to official grades, it can be from 0 (no DR) to 4 (Proliferative DR)
- DME_level: it indicates the presence or not of DME
- gradability: it indicates if the image has quality enough (0 - insufficient quality / ungradable, 1 - gradable)
- old_size_x: it's the width of the original image, in pixels, before cropping and resizing
- old_size_y: it's the height of the original image, in pixels, before cropping and resizing

'old_size_x' and 'old_size_y' will be used to discard images that may have been upscaled too much and perhaps they have lossed quality

No info about FOV is needed because all images have been processed (cropped, resized and padded)

As previous CSV file, it is important to note that if any image does not have any information about any of the above fields, those fields will be 
filled with -1. So, -1 will mean unknown value.

Use example: > python ./a02_redistribute_datasets.py -d eyepacs messidor_2 aptos_2019 --size 540
'''


def process_images(data, dataset_new_path, img_size, gradable):
    '''
    This function receives a DataFrame object (data) which must be have these columns: image, path, DR_level, DME_level, gradability, cntr_center_x, cntr_center_y.
    Reads row by row the DataFrame and copies the information about each image.
    Process the image by calling the function process_image in lib.preprocess. It will crop and resize the image using its radius, contour center and the new size given as parameter
    Returns five lists, in a tuple, which are: size of image in x and y dimmensions, radius of the detected contour and the x and y coordinates of the contour center
    (size_x, size_y, radius, pos_center_x, pos_center_y)
    '''
    # Information needed
    images = []
    paths = []
    dr_lbls = []
    dme_lbls = []
    grad_lbls = []
    old_size_x = []
    old_size_y = []

    for row in data.itertuples():
        # Save information of image
        images.append(row.image)
        
        dr_lbls.append(int(row.DR_level))
        dme_lbls.append(int(row.DME_level))
        grad_lbls.append(int(row.gradability))
        old_size_x.append(int(row.cntr_center_x))
        old_size_y.append(int(row.cntr_center_y))

        folder = str(int(row.DR_level)) if gradable else 'ungradables'

        # Process image and get path
        new_path = process_image(img_name=row.image, 
                                 img_path=row.path, 
                                 save_path=dataset_new_path + folder + '/', 
                                 radius=row.cntr_radius, 
                                 c_x=row.cntr_center_x, 
                                 c_y=row.cntr_center_y, 
                                 size=img_size)

        paths.append(new_path)
    
    return (images, paths, dr_lbls, dme_lbls, grad_lbls, old_size_x, old_size_y)

# Define arguments
parser = argparse.ArgumentParser(description='Process all images of specified datasets and distribute them into new directories inside data/processed_datasets/')

parser.add_argument('-d','--datasets',
                    nargs='+',
                    help='Specify which datasets have to be processed and distributed',
                    default=['aptos_2019', 'eyepacs', 'messidor', 'messidor_2_abramoff', 'messidor_2'])

parser.add_argument('-s','--size', type=int, nargs=1, help='Specify the size wanted to resize all images once the have been cropped', default=[540])

# Read arguments
args = parser.parse_args()

datasets = args.datasets
IMG_SIZE = args.size[0]

def main():

    # datasets = ['aptos_2019', 'eyepacs', 'messidor', 'messidor_2_abramoff', 'messidor_2']

    # Create the directory where all datasets will be saved
    if not os.path.exists('data/processed_datasets'):
        os.mkdir('data/processed_datasets')

    # Creates the directory where there will be a copy of all datasets' general .csv file
    if not os.path.exists('data/processed_datasets/0_csvs'):
        os.mkdir('data/processed_datasets/0_csvs')

    # Create all new directories for each dataset and process its images
    for d in datasets:
        dataset = pd.read_csv('data/original_datasets/' + d + '/' + d + '.csv')

        dataset_new_path = 'data/processed_datasets/' + d +'/'

        # Check if this dataset has a directory created. If this is true, it means that this dataset was processed previously
        if os.path.exists(dataset_new_path):
            print('Dataset',d,'is already processed')
            continue

        # Create a new directory for current dataset (d)
        os.mkdir(dataset_new_path)

        # Get all DR levels and gradability labels
        gradability_lb = list(dataset['gradability'].unique())
        dr_levels = list(dataset['DR_level'].unique())
        # Convert all DR levels to int because they will be converted to string to create new folders for each level
        # We do not want float numbers as folder names
        dr_levels = list(map(int, dr_levels))

        # Check if there are gradability labels. If there is only one gradability label (-1), it means that no image was labeled for gradability
        if len(gradability_lb) == 1 and gradability_lb[0] == -1:
            print('Dataset',d,'does not have gradability labels')

            # Create just an empty file with the aim of informing the user
            aux_file = dataset_new_path + 'This dataset does not have gradability labels.txt'
            if not os.path.exists(aux_file):
                open(aux_file, 'a').close()
        else:
            # Create directory for ungradable images if label 0 is present in gradability labels 
            if 0 in gradability_lb and not os.path.exists(dataset_new_path + 'ungradables'):
                os.mkdir(dataset_new_path + 'ungradables')
        
        # Create directories for each DR level in this dataset (except -1 level, which means unknown DR level)
        for dr in dr_levels:
            if dr != -1 and not os.path.exists(dataset_new_path + str(dr)):
                # Create directory for each DR level, named as its DR level assigned
                os.mkdir(dataset_new_path + str(dr))

        # Redistribute and process images
        
        # Check if there are images whose FOV was not detected (radius == -1)
        # These images will be copied to the 'no_FOV_detected' and ignored in the .csv file
        # Find the minimum value in radius column, if it is -1, it means that there are some images without FOV
        if dataset['cntr_radius'].min() == -1:
            no_FOV_dir = dataset_new_path + 'no_FOV_detected'
            # Create directory for undetected FOV images
            if not os.path.exists(no_FOV_dir):
                os.mkdir(no_FOV_dir)

            # Sort all data by radius. With this, all images without FOV will appear first
            dataset.sort_values(by='cntr_radius',inplace=True,ignore_index=True)

            # Copy all these undetected FOV images to the new directory
            count = 0
            for row in dataset.itertuples():
                if row.cntr_radius == -1:
                    copy2(row.path, no_FOV_dir)
                    count += 1
                else:
                    # With dataset sorted by radius, and undetected FOV images has radius equals to -1, these images will be placed before 'FOV detected' images
                    # When the actual image's radius isn't '-1', it stops
                    break
            
            print('Dataset',d,'has',count,'images whose FOV was not detected --- Saved these images in:',no_FOV_dir)

            # Delete from dataset -- There images are useless, they cannot be processed
            dataset = pd.DataFrame(dataset.tail(dataset.shape[0]-count))

        # Information for new csv
        # Create an empty list for each field in the new .csv file
        images = []
        paths = []
        dr_lbls = []
        dme_lbls = []
        grad_lbls = []
        old_size_x = []
        old_size_y = []
        # Radius and center of FOV are not necessary, images will have been cropped and resize with the specified size

        # If the ungradable label (0) is present in this dataset, procees those images first
        if 0 in gradability_lb:
            # Sort by gradability -- Ungradable images will be placed first
            dataset.sort_values(by='gradability', inplace=True, ignore_index=True)

            count = 0
            for row in dataset.itertuples():
                if row.gradability == 0:
                    count += 1
                else:
                    # With dataset sorted by gradability, and 'ungradable' label being 0, these images will be placed before 'gradable' images
                    # When the actual image's gradability isn't '0', it stops
                    break

            # Process ungradable images in parallel 
            num_process = 6

            # These are all ungradable images
            ungradables_dataset = dataset.head(count)

            # Split in 'num_process' chunks
            chunks_datasets = np.array_split(ungradables_dataset, num_process)

            # Create Process Pool
            with futures.ProcessPoolExecutor(max_workers=num_process) as ex:
                    futures_list = []
                    for chunk in chunks_datasets:
                        # Start a process
                        futures_list.append(ex.submit(process_images, chunk, dataset_new_path, IMG_SIZE, False))

                    for process in futures_list:
                        while process.running():
                            pass
                        # Get info of all images in this chunk
                        (images_tmp, paths_tmp, dr_lbls_tmp, dme_lbls_tmp, grad_lbls_tmp, old_size_x_tmp, old_size_y_tmp) = process.result()

                        # Add to lists
                        images.extend(images_tmp)
                        paths.extend(paths_tmp)
                        dr_lbls.extend(dr_lbls_tmp)
                        dme_lbls.extend(dme_lbls_tmp)
                        grad_lbls.extend(grad_lbls_tmp)
                        old_size_x.extend(old_size_x_tmp)
                        old_size_y.extend(old_size_y_tmp)
            
            print('Dataset',d,'has',count,'ungradable images --- Saved these images in:',dataset_new_path + 'ungradables')

            # Delete from dataset -- ungradable images have already been processed
            dataset = pd.DataFrame(dataset.tail(dataset.shape[0]-count))

        # Now it is turn to process gradable images. These images will be saved in their DR level folder

        # Process gradable images in parallel 
        num_process = 6

        # Split in 'num_process' chunks
        chunks_datasets = np.array_split(dataset, num_process)

        # Create Process Pool
        with futures.ProcessPoolExecutor(max_workers=num_process) as ex:
                futures_list = []
                for chunk in chunks_datasets:
                    # Start a process
                    futures_list.append(ex.submit(process_images, chunk, dataset_new_path, IMG_SIZE, True))

                for process in futures_list:
                    while process.running():
                        pass
                    # Get info of all images in this chunk
                    (images_tmp, paths_tmp, dr_lbls_tmp, dme_lbls_tmp, grad_lbls_tmp, old_size_x_tmp, old_size_y_tmp) = process.result()

                    # Add to lists
                    images.extend(images_tmp)
                    paths.extend(paths_tmp)
                    dr_lbls.extend(dr_lbls_tmp)
                    dme_lbls.extend(dme_lbls_tmp)
                    grad_lbls.extend(grad_lbls_tmp)
                    old_size_x.extend(old_size_x_tmp)
                    old_size_y.extend(old_size_y_tmp)

        # Generate the new .csv file, with same structure for all datasets
        new_csv = pd.DataFrame({
            'image' : images,
            'path' : paths,
            'DR_level' : dr_lbls,
            'DME_level' : dme_lbls,
            'gradability' : grad_lbls,
            'old_size_x' : old_size_x,
            'old_size_y' : old_size_y
        })

        # Save new csv file
        new_csv.to_csv(dataset_new_path + d +'.csv', index=False)

        print('Saved new csv for',d,'dataset in:',dataset_new_path + d +'.csv') 

        # Create a copy of the .csv file to the 0_csvs/ directory
        copy2(dataset_new_path + d +'.csv', 'data/processed_datasets/0_csvs')

if __name__ == "__main__":
    main()

    # There is a little chance that some images could have been cropped with an incorrect shape
    # Probably, this might have been result of any unespected value in a round operation

    print('Checking that all images have the correct size...')

    for d in datasets:
        dataframe = pd.read_csv('data/processed_datasets/0_csvs/' + d + '.csv')

        for row in dataframe.itertuples():
            img = cv2.imread(row.path)
            h, w, _ = img.shape
            if h != IMG_SIZE or w != IMG_SIZE:
                img_new = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                img_new[:h, :w, :] = img[:IMG_SIZE, :IMG_SIZE, :]
                os.remove(row.path)
                cv2.imwrite(row.path, img_new)
                print('Image:',row.path,'had this shape',img.shape,'Corrected to:',(IMG_SIZE,IMG_SIZE,3))