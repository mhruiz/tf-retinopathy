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

    # Parse arguments: only expected argument --> dataset_dir

    parser = argparse.ArgumentParser(description='Prepare Messidor 2 Abramoff data set.')
    parser.add_argument("--dataset_dir", 
                        help="Directory where Messidor 2 Abramoff resides.",
                        default="data/original_datasets/messidor_2_abramoff")
                        
    args = parser.parse_args()

    # Get directory where this dataset resides
    dataset_dir = str(args.dataset_dir)

    # Read csv file with all known information about the images: image name (examid) and DR level (rDR: 0 or 1)
    # This dataset do not provide DR level labels, it provides information about presence of referable DR (DR levels 2, 3 and 4)
    all_info = pd.read_csv(join(dataset_dir, 'abramoff-messidor-2-refstandard-jul16.csv'))
    
    # Create empty lists for known fields
    image_names = []
    image_paths = []
    DR_levels = []

    # Fill these lists
    for row in all_info.itertuples():
        base_name = row.examid
        dr_level = row.rDR

        # All images of this dataset do not have the same file extension
        # It's necessary to find all files whose name starts with 'base_name'
        files = glob(dataset_dir + '/images/' + base_name + '*')

        # This only works in Windows: replace '\\' (Windows directory separator) with '/' for a linux friendly path form
        files = list(map(lambda x: x.replace('\\', '/'), files))

        # For any file found whose name starts with 'base_name'
        for file in files:
            # Save only its full name, without file extension 
            # Split with '/' and take last --> image_name.extension
            # Split with '.' and taken first --> image_name
            image_names.append(''.join(file.split('/')[-1].split('.')[:-1]))
            # Get full path
            image_paths.append(file)
            # Get DR evidence
            DR_levels.append(dr_level)

    # Get info necessary from images: size of image and contour (center and radius)
    size_x = []
    size_y = []
    pos_center_x = []
    pos_center_y = []
    radius = []

    # Process in parallel
    num_process = 6

    # Split in 'num_process' chunks the list od paths
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
    messidor_2_abramoff_csv = pd.DataFrame({
        'image' : image_names,
        'path' : image_paths,
        'DR_level' : DR_levels,
        'DME_level' : [-1 for i in range(len(DR_levels))], # There is no information about evidence DME, so its value will be -1
        'gradability' : [-1 for i in range(len(DR_levels))], # There is no information about gradability, so its value will be -1
        'size_x' : size_x,
        'size_y' : size_y,
        'cntr_radius' : radius,
        'cntr_center_x' : pos_center_x,
        'cntr_center_y' : pos_center_y
    })

    print('Showing csv generated')
    print(messidor_2_abramoff_csv.head())
    print('....')
    print(messidor_2_abramoff_csv.tail())
    print('')
            
    # Save new .csv file
    messidor_2_abramoff_csv.to_csv(dataset_dir + '/messidor_2_abramoff.csv', index=False)

    print('Saved messidor_2_abramoff.csv at:',dataset_dir + '/messidor_2_abramoff.csv')

if __name__ == '__main__':
    main()