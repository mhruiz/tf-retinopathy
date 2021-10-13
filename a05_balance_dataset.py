import pandas as pd
import numpy as np
import argparse

'''
This script allows the user to create a new csv file which will contain the same number of images per class.
The csv file must have at least these columns: 'path' and 'DR_level'. If the given csv file was generated with another script,
it may have a third column named 'label', which will be saved too, but ignored.
'''

# Define arguments
parser = argparse.ArgumentParser(description='Create a balanced CSV file from an existing one')

parser.add_argument('dataset',
                    type=str,
                    nargs='+',
                    help='Specify the path to the csv file that will be used to create the balanced dataset')

parser.add_argument('-c', '--classes', 
                    nargs='+', 
                    help='Give the DR Levels for each class. For example, to define 2 classes (no DR vs DR): 0 1234', 
                    default=['0', '1234'])

parser.add_argument('--name',
                    nargs=1,
                    type=str,
                    help='Name for new csv file')

# Read arguments

args = parser.parse_args()

# Read arguments ------------------------------------
dataset_name = args.dataset[0]

# Convert classes arguments to list of lists of integers
classes = [[int(j) for j in i] for i in args.classes]

# Get all DR levels specified in a single flatten list
specified_DR_levels = [j for i in classes for j in i]

# Read csv file
dataset = pd.read_csv(dataset_name)

# Get a list of all DR levels
existing_DR_levels = dataset['DR_level'].unique().tolist()

# Check that every given DR level exists on this csv file
assert all(i in existing_DR_levels for i in specified_DR_levels), 'There is at least one DR level specified that is not present on this csv file. \n' + \
    '  Current DR levels on this csv file: ' + ', '.join(list(map(str,existing_DR_levels))) + '\n' + \
        '  Specified DR levels: ' + ', '.join(list(map(str,specified_DR_levels)))

# Create a dictionary with an entry for each DR level specified
splitted_datasets = {i: dataset.loc[dataset.DR_level == i] for i in specified_DR_levels}

# Join DR levels following the desired distribution
distributed_datasets = {}
for i, c in enumerate(classes):
    for dr_lvl in c:
        if distributed_datasets.get(i) is None:
            distributed_datasets[i] = splitted_datasets[dr_lvl]
        else:
            distributed_datasets[i] = pd.concat([distributed_datasets[i], splitted_datasets[dr_lvl]],ignore_index=True)

# Get the minority class
smallest_size = np.Inf
for d in distributed_datasets:
    # Shuffle each class
    distributed_datasets[d] = distributed_datasets[d].sample(frac=1, random_state=2021)

    if smallest_size > distributed_datasets[d].shape[0]:
        smallest_size = distributed_datasets[d].shape[0]

balanced_dataset = distributed_datasets[0].head(smallest_size).copy()

for i in range(1, len(classes)):
    balanced_dataset = pd.concat([balanced_dataset, distributed_datasets[i].head(smallest_size)], ignore_index=True)

# Perform a shuffle before saving the balanced csv file
balanced_dataset = balanced_dataset.sample(frac=1, random_state=2021)

if args.name is None:
    balanced_dataset.to_csv(dataset_name.split('.')[0] + '_BALANCED.csv', index=False)
else:
    balanced_dataset.to_csv(args.name[0] + '.csv', index=False)
