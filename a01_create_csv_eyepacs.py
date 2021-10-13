from concurrent import futures
from lib import preprocess
from os.path import join
import pandas as pd
import numpy as np
import argparse
import sys

def main():

    # Parse arguments: only expected argument --> dataset_dir

    parser = argparse.ArgumentParser(description='Prepare EyePACS data set.')
    parser.add_argument("--dataset_dir", 
                        help="Directory where EyePACS resides.",
                        default="data/original_datasets/eyepacs")
                        
    args = parser.parse_args()

    # Get directory where this dataset resides
    dataset_dir = str(args.dataset_dir)

    # Specify location for train and test images. They are in different directories
    data_train_dir = dataset_dir + '/images/train/'
    data_test_dir = dataset_dir + '/images/test/'

    # Get full paths to .csv files (DR labels and Gradability)
    train_labels = join(dataset_dir, 'trainLabels.csv')
    test_labels = join(dataset_dir, 'testLabels.csv')
    gradability_grades = join(dataset_dir, 'eyepacs_gradability_grades.csv')

    # Read DR level CSVs
    train_lb = pd.read_csv(train_labels, dtype={'image':str,'level':'int8'})
    test_lb = pd.read_csv(test_labels, dtype={'image':str,'level':'int8','Usage':str})
    
    # Test csv has a column named 'usage' which is not needed
    # Delete this column making both DataFrames have these columns: image, level
    del test_lb['Usage']

    # Commented code was used to check that all images were .jpeg
    # import os
    # dicc = {}
    # for i in os.listdir(data_train_dir):
    #     extension = i.split('.')[1]
    #     if dicc.get(extension) is None:
    #         dicc[extension] = 1
    #     else:
    #         dicc[extension] = dicc[extension] + 1
    # print(dicc) # Checked - all images are .jpeg
    # dicc = {}
    # for i in os.listdir(data_test_dir):
    #     extension = i.split('.')[1]
    #     if dicc.get(extension) is None:
    #         dicc[extension] = 1
    #     else:
    #         dicc[extension] = dicc[extension] + 1
    # print(dicc) # Checked - all images are .jpeg
    # sys.exit(0)


    def create_path_column(dir, column):
        '''
        This function receives a directory and a column of a DataFrame which contains all image names.
        Returns a list which will contain the path to all images in 'column'
        '''
        path = []
        for i in column:
            img_path = dir + i + '.jpeg' # Checked - all images are .jpeg
            path.append(img_path)
        return path

    # Add new column to train labels: path --> data/eyepacs/images/train/
    path = create_path_column(data_train_dir, train_lb['image'])
    train_lb.insert(1, 'path', path)

    # Add new column to test labels: path --> data/eyepacs/images/test/
    path = create_path_column(data_test_dir, test_lb['image'])
    test_lb.insert(1, 'path', path)
    # Both DataFrames must have 'path' column in second position, so they will have the same structure and we will be able to concat them

    # Now we can join all DR labels from train and test (thanks to they have the same structure)
    all_DR_labels = pd.concat([train_lb, test_lb],ignore_index=True)

    # Read csv with gradability grades
    gradable_lb = pd.read_csv(gradability_grades, delimiter=' ', dtype={'image_name':str,'gradability':'int8'})

    # Check if there are the same number of rows
    if all_DR_labels.shape[0] != gradable_lb.shape[0]:
        print("DR Level csv and Gradability csv don't have the same number of rows")
        sys.exit(-1)

    # Sort by name
    all_DR_labels.sort_values(by='image',inplace=True,ignore_index=True)
    gradable_lb.sort_values(by='image_name',inplace=True,ignore_index=True)
    # By sorting them using the image name field, the can be joined after (when creatijg the new csv)

    # Check if both DataFrames have any difference in image name
    # They should have same order
    for row1, row2 in zip(all_DR_labels.itertuples(), gradable_lb.itertuples()):
        if (row1.image != row2.image_name):
            print("CSVs don't have the same image names")
            sys.exit(-1)

    # Get info necessary from images: size of image and contour (center and radius)
    size_x = []
    size_y = []
    pos_center_x = [] # Of the contour
    pos_center_y = []
    radius = []  # Of the contour

    # Get only image paths
    only_paths = pd.DataFrame(all_DR_labels['path'])

    # Process in parallel
    num_process = 6

    # Split in 'num_process' chunks
    chunk_paths = np.array_split(only_paths, num_process)

    # Create Process Pool for 6 -- 6h
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

    # # Secuential --- should takes 17h-18h to finish
    # for row1 in all_DR_labels.itertuples():
    #     # Read image
    #     img = cv2.imread(row1.path, -1)

    #     # Image size
    #     s_y = img.shape[0]
    #     s_x = img.shape[1]
    #     size_x.append(s_x)
    #     size_y.append(s_y)

    #     # Check if contour is detected
    #     contour = preprocess._find_contours(img)
    #     if contour is None:
    #         # If no contour has been detected, write -1
    #         pos_center_x.append(-1)
    #         pos_center_y.append(-1)
    #         radius.append(-1)
    #     else:
    #         center, r = contour
    #         c_x, c_y = center
    #         pos_center_x.append(c_x)
    #         pos_center_y.append(c_y)
    #         radius.append(r)

    # Generate the new .csv file, with same structure for all datasets
    eyepacs_csv = pd.DataFrame({
        'image' : all_DR_labels['image'],
        'path' : all_DR_labels['path'],
        'DR_level' : all_DR_labels['level'],
        'DME_level' : [-1 for i in range(len(all_DR_labels['image']))], # -1 because there is no information about DME evidence
        'gradability' : gradable_lb['gradability'],
        'size_x' : size_x,
        'size_y' : size_y,
        'cntr_radius' : radius,
        'cntr_center_x' : pos_center_x,
        'cntr_center_y' : pos_center_y
    })

    print('Showing csv generated')
    print(eyepacs_csv.head())
    print('....')
    print(eyepacs_csv.tail())
    print('')

    # Save new .csv file
    eyepacs_csv.to_csv(dataset_dir + '/eyepacs.csv', index=False)

    print('Saved eyepacs.csv at:',dataset_dir + '/eyepacs.csv')

if __name__ == "__main__":
    main()