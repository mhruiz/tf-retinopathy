from concurrent import futures
from lib import preprocess
from os.path import join
from glob import glob
import pandas as pd
import numpy as np
import argparse
import sys
import os

def main():

    # Parse arguments: only expected one argument --> dataset_dir

    parser = argparse.ArgumentParser(description='Prepare Aptos 2019 data set.')
    parser.add_argument("--dataset_dir", 
                        help="Directory where Aptos 2019 resides.",
                        default="data/original_datasets/aptos_2019")
                        
    args = parser.parse_args()

    # Get directory where this dataset resides
    dataset_dir = str(args.dataset_dir)

    # Read the .csv file which contains image names and their DR level
    all_info = pd.read_csv(join(dataset_dir, 'train.csv'))

    # Creates a list of paths to all the images 
    # This list will be used to generate the general csv
    image_paths = []
    for row in all_info.itertuples():
        image_paths.append(dataset_dir + '/images/' + row.id_code + '.png') # Checked - all images are .png

    # Get info necessary from images: size of image and contour (center and radius)
    size_x = []
    size_y = []
    pos_center_x = [] # Of the contour
    pos_center_y = []
    radius = []  # Of the contour

    # Process in parallel
    num_process = 6

    # Split in 'num_process' chunks the list of paths
    # First, create a pandas DataFrame with this list
    only_path = pd.DataFrame({'path' : image_paths})
    # Then, split it
    chunk_paths = np.array_split(only_path, num_process)

    # Create Process Pool for 6
    with futures.ProcessPoolExecutor(max_workers=num_process) as ex:
            futures_list = []
            for chunk in chunk_paths:
                # Start a process
                futures_list.append(ex.submit(preprocess.get_info_all_images_chunk, chunk))

            # For each process, get its output
            for process in futures_list:
                while process.running():
                    pass
                # Get info of all images in this chunk
                (size_x_tmp, size_y_tmp, radius_tmp, pos_center_x_tmp, pos_center_y_tmp) = process.result()
                # Add to lists
                size_x.extend(size_x_tmp)
                size_y.extend(size_y_tmp)
                radius.extend(radius_tmp)
                pos_center_x.extend(pos_center_x_tmp)
                pos_center_y.extend(pos_center_y_tmp)

    # Generate the new .csv file, with same structure for all datasets
    aptos_2019_csv = pd.DataFrame({
        'image' : all_info['id_code'],
        'path' : image_paths,
        'DR_level' : all_info['diagnosis'],
        'DME_level' : [-1 for i in range(len(image_paths))], # There is no information about evidence DME, so its value will be -1
        'gradability' : [-1 for i in range(len(image_paths))], # There is no information about gradability, so its value will be -1
        'size_x' : size_x,
        'size_y' : size_y,
        'cntr_radius' : radius,
        'cntr_center_x' : pos_center_x,
        'cntr_center_y' : pos_center_y
    })

    print('Showing csv generated')
    print(aptos_2019_csv.head())
    print('....')
    print(aptos_2019_csv.tail())
    print('')
            
    # Save new .csv file
    aptos_2019_csv.to_csv(dataset_dir + '/aptos_2019.csv', index=False)

    print('Saved aptos_2019.csv at:',dataset_dir + '/aptos_2019.csv')

if __name__ == '__main__':
    main()