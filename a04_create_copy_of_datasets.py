from shutil import copy2, rmtree
from glob import glob
from numpy.core.fromnumeric import partition
import pandas as pd
import numpy as np
import argparse
import sys
import os

'''
This script allows the user to create a copy of all images spcified in the given dataset and rewriting its csv files.
This can be useful when it is necessary to take some images and move them to another directory, without taking all
the 97.000 images.

The given dataset must have been obtenined by calling script 'a03_define_custom_datasets.py'.
'''

# Define arguments
parser = argparse.ArgumentParser(description='Create a copy of given dataset')

parser.add_argument('name',
                    nargs=1,
                    type=str,
                    help='Specify which dataset will be used to create a copy of its images')

parser.add_argument('--new_path',
                    nargs=1,
                    type=str,
                    help='Specify the new folder to store all copied images')

args = parser.parse_args()

# This dataset must be copied
dataset_name = args.name[0]

# This is the main csv file with this name
dataset_file = glob(dataset_name + '-ALL.csv')

if not dataset_file: # empty list -> no csv file found
    print('Dataset \'' + dataset_name + '\' does not exist. There must be a csv called: \'' + dataset_name + '-ALL.csv\'')
    sys.exit(0)

dataset_file = dataset_file[0]

# Read dataset
dataset = pd.read_csv(dataset_file)

# Ensure the user wants to copy a huge number of images
choice = input('This dataset has ' + str(dataset.shape[0]) + ' images. It could take too much time. Do you want to copy all its images? [y/n]: ')

if choice != 'y' and choice != 'Y':
    sys.exit(0)

# Prepare new directory 

new_dir = args.new_path

if new_dir is not None:
    new_dir = new_dir[0]

    # If the user has given a directory name that already exists, it will not bet erased
    if os.path.exists(new_dir):
        print('The given directory \'' + new_dir + '\' already exists. Please provide another directory name or delete first that folder')
        sys.exit(0)

else:
    new_dir = 'dataset_copy'

    # If no directory was given, default directory will be used. If this directory already exists, it will be deleted
    if os.path.exists(new_dir):
        print('Removing', new_dir)
        rmtree(new_dir)

# Create a copy of all images
os.mkdir(new_dir)

new_paths = []

print('Copying images')

fifth_percentage = dataset.shape[0] // 20

for i, row in enumerate(dataset.itertuples()):
    # Get original full path
    path = row.path

    # Original dataset
    original_dataset_name = path.split('/')[2]
    # Image name
    image_name = path.split('/')[-1]

    # New path
    new_path_image = new_dir + '/' + original_dataset_name

    if not os.path.exists(new_path_image):
        os.mkdir(new_path_image)

    # Copy image
    copy2(path, new_path_image)

    # Save new path
    new_paths.append(new_path_image + '/' + image_name)

    if i!= 0 and i % fifth_percentage == 0:
        print('  ' + str((i // fifth_percentage) * 5) + '%' + ' completed', end='\r')

print('All images were copied')

# New csv file
new_dataset = pd.DataFrame({
    'path': new_paths,
    'label': dataset['label'],
    'DR_level': dataset['DR_level']
})

if os.path.exists('copy_' + dataset_name + '-ALL.csv'):
    os.remove('copy_' + dataset_name + '-ALL.csv')

new_dataset.to_csv('copy_' + dataset_name + '-ALL.csv', index=False)

# Create modified copy for TRAIN, VALIDATION and TEST csv files, if they exists

for dt in ['TRAIN', 'VALIDATION', 'TEST']:
    # Get file name
    dataset_file = glob(dataset_name + '-' + dt + '*.csv')

    # Check if exists
    if not dataset_file: # empty list
        continue
    
    # Get file name
    dataset_file = dataset_file[0]

    # Read dataset
    dataset = pd.read_csv(dataset_file)

    new_paths = []
    for row in dataset.itertuples():
        # Original dataset
        original_dataset_name = row.path.split('/')[2]
        # Image name
        image_name = row.path.split('/')[-1]

        # New path
        new_path_image = new_dir + '/' + original_dataset_name + '/' + image_name

        # Save new path
        new_paths.append(new_path_image)

    # New csv file
    new_dataset = pd.DataFrame({
        'path': new_paths,
        'label': dataset['label'],
        'DR_level': dataset['DR_level']
    })

    if os.path.exists('copy_' + dataset_name + '-' + dt + '.csv'):
        os.remove('copy_' + dataset_name + '-' + dt + '.csv')

    new_dataset.to_csv('copy_' + dataset_name + '-' + dt + '.csv', index=False)