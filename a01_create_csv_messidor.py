from concurrent import futures
from lib import preprocess
from os.path import join
from glob import glob
import pandas as pd
import numpy as np
import argparse
import xlrd
import sys
import os

def main():

    # Parse arguments: only expected argument --> dataset_dir

    parser = argparse.ArgumentParser(description='Prepare Messidor data set.')
    parser.add_argument("--dataset_dir", 
                        help="Directory where Messidor resides.",
                        default="data/original_datasets/messidor")
                        
    args = parser.parse_args()

    # Get directory where this dataset resides
    dataset_dir = str(args.dataset_dir)

    # These images are duplicated
    # 16 August 2017: Image duplicates in Base33
    duplicates = ["20051202_54744_0400_PP", "20051202_40508_0400_PP",
                  "20051202_41238_0400_PP", "20051202_41260_0400_PP",
                  "20051202_54530_0400_PP", "20051205_33025_0400_PP",
                  "20051202_55607_0400_PP", "20051202_41034_0400_PP",
                  "20051205_35099_0400_PP", "20051202_54555_0400_PP",
                  "20051205_35110_0400_PP", "20051202_54611_0400_PP",
                  "20051202_55498_0400_PP"]

    # COMMENTED: It's not necessary to remove them, they won't be part of the .csv file
    # Remove all duplicated images
    # for duplicate_img in duplicates:
    #     full_path = dataset_dir + '/' + duplicate_img + '.tif'
    #     if os.path.exists(full_path):
    #         os.remove(full_path)
    
    image_names = []
    image_paths = []
    DR_levels = []
    DME_levels = []

    # Get image names, DR and DME grades from annotation files (.xls)
    annotation_files = glob(dataset_dir + '/*.xls')
    # For each file
    for annotation_file in annotation_files:
        workbook = xlrd.open_workbook(annotation_file)
        worksheet = workbook.sheet_by_index(0)

        # For each row
        for row in range(1, worksheet.nrows):
            # Get image name withput file extension
            image_names.append(str(worksheet.cell(row, 0).value).split('.')[0])
            # Get full path
            image_paths.append(dataset_dir + '/images/' + str(worksheet.cell(row, 0).value))
            # Get DR level
            DR_levels.append(int(worksheet.cell(row, 2).value))
            # Get DME evidence
            DME_levels.append(int(worksheet.cell(row, 3).value))

    # Correct erratums: some images were missclassified
    # Create a dictionary where each missclassified image will have its correct DR level
    erratums = {}

    # 31 August 2016: Erratum in Base11 Excel file
    erratums["20051020_63045_0100_PP"] = 0

    # 24 October 2016: Erratum in Base11 and Base 13 Excel files
    erratums["20051020_64007_0100_PP"] = 3
    erratums["20051020_63936_0100_PP"] = 1
    erratums["20060523_48477_0100_PP"] = 3

    count = 0
    for i in range(len(image_names)):
        # Check if image is in the erratums dictionary
        if erratums.get(image_names[i]) is not None:
            # Correct DR level
            DR_levels[i] = erratums[image_names[i]]
            count += 1
            # When all wrong images were corrected, break loop
            if count == len(erratums):
                break

    # Read gradability information
    gradability_lb = pd.read_csv(join(dataset_dir, 'messidor_gradability_grades.csv'), delimiter=' ', dtype={'image_name':str,'gradability':'int8'})

    # Create a temporal DataFrame with image name, path, DR level and DME evidence
    all_info = pd.DataFrame({
        'image' : image_names,
        'path' : image_paths,
        'DR_level' : DR_levels,
        'DME_level' : DME_levels
    })

    # Delete from 'all_info' all duplicated images 
    positions = []
    for row in all_info.itertuples():
        try:
            # Check if this image is duplicated (inside duplicates list)
            duplicates.index(row.image)
            # Save position of index
            positions.append(row.Index)
            if len(positions) == len(duplicates):
                break
        except ValueError:
            # If this image is not in duplicated list, it will throws a ValueError
            pass
    
    # Delete all duplicated images
    all_info.drop(positions, axis=0,inplace=True)

    # Check if there are the same number of rows in DR Level and Gradability DataFrames
    if all_info.shape[0] != gradability_lb.shape[0]:
        print("DR Level csv and Gradability csv don't have the same number of rows")
        sys.exit(-1)

    # Sort by name
    all_info.sort_values(by='image',inplace=True,ignore_index=True)
    gradability_lb.sort_values(by='image_name',inplace=True,ignore_index=True)
    # By sorting them using the image name field, the can be joined after (when creatijg the new csv)

    # Check if both DataFrames have any difference in image name
    # They should have same image names
    for row1, row2 in zip(all_info.itertuples(), gradability_lb.itertuples()):
        if (row1.image != row2.image_name):
            print("CSVs don't have the same image names")
            sys.exit(-1)

    # Get info necessary from images: size of image and contour (center and radius)
    size_x = []
    size_y = []
    pos_center_x = []
    pos_center_y = []
    radius = []

    # Process in parallel
    num_process = 6

    # Split in 'num_process' chunks the list of paths
    # First, create a pandas DataFrame with this column
    only_path = pd.DataFrame(all_info['path'])
    # Then, split it
    chunk_paths = np.array_split(only_path, num_process)

    # Create Process Pool for 6
    with futures.ProcessPoolExecutor(max_workers=num_process) as ex:
            futures_list = []
            for chunk in chunk_paths:
                # Start process
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
    messidor_csv = pd.DataFrame({
        'image' : all_info['image'],
        'path' : all_info['path'],
        'DR_level' : all_info['DR_level'],
        'DME_level' : all_info['DME_level'],
        'gradability' : gradability_lb['gradability'],
        'size_x' : size_x,
        'size_y' : size_y,
        'cntr_radius' : radius,
        'cntr_center_x' : pos_center_x,
        'cntr_center_y' : pos_center_y
    })

    print('Showing csv generated')
    print(messidor_csv.head())
    print('....')
    print(messidor_csv.tail())
    print('')
            
    # Save new .csv file
    messidor_csv.to_csv(dataset_dir + '/messidor.csv', index=False)

    print('Saved messidor.csv at:',dataset_dir + '/messidor.csv')

if __name__ == '__main__':
    main()