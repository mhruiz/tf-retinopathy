from concurrent import futures
from lib import preprocess
from os.path import join
import pandas as pd
import numpy as np
import argparse
import sys
import os

def main():

    # Parse arguments: only expected argument --> dataset_dir

    parser = argparse.ArgumentParser(description='Prepare Messidor 2 data set.')
    parser.add_argument("--dataset_dir", 
                        help="Directory where Messidor 2 resides.",
                        default="data/original_datasets/messidor_2")
                        
    args = parser.parse_args()

    # Get directory where this dataset resides
    dataset_dir = str(args.dataset_dir)

    # Read the .csv file which contains image names (with file extension), DR levels, DME levels and grabability labels
    all_info = pd.read_csv(join(dataset_dir, 'messidor_data_adjudication.csv'))
    
    # Create empty lists
    image_names = []
    image_paths = []

    for row in all_info.itertuples():
        # Get only image name, without file extension
        image_names.append(row.image_id.split('.')[0])
        # Get full path of image (knowing it's saved inside 'images' directory)
        image_paths.append(dataset_dir + '/images/' + row.image_id)

    # Delete unnecessary column
    all_info.drop('image_id', axis=1, inplace=True)

    # All non-gradable images has no DR or DME level adjudicated
    # So it's necessary to replace NaN elements with -1 (unknown value)
    all_info.replace(np.NaN, -1, inplace=True)

    # Get info necessary from images: size of image and contour (center and radius)
    size_x = []
    size_y = []
    pos_center_x = []
    pos_center_y = []
    radius = []

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

            for process in futures_list:
                # Wait
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
    messidor_2_csv = pd.DataFrame({
        'image' : image_names,
        'path' : image_paths,
        'DR_level' : all_info['adjudicated_dr_grade'],
        'DME_level' : all_info['adjudicated_dme'],
        'gradability' : all_info['adjudicated_gradable'],
        'size_x' : size_x,
        'size_y' : size_y,
        'cntr_radius' : radius,
        'cntr_center_x' : pos_center_x,
        'cntr_center_y' : pos_center_y
    })

    print('Showing csv generated')
    print(messidor_2_csv.head())
    print('....')
    print(messidor_2_csv.tail())
    print('')

    # Save new .csv file
    messidor_2_csv.to_csv(dataset_dir + '/messidor_2.csv', index=False)

    print('Saved messidor_2.csv at:',dataset_dir + '/messidor_2.csv')

if __name__ == '__main__':
    main()