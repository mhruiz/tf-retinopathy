import pandas as pd
import numpy as np
import argparse
import sys
import os

'''
This script allows the user to create training, validation and test datasets by combining existing datasets
+---------------------------------------------------------------------------------------------------------------------+
| (all datasets must have been processed and stored in 'data/processed_datasets/' with a02_redistribute_datasets.py), |
+---------------------------------------------------------------------------------------------------------------------+
with the classes and the proportion established by the user. In addition, you can choose other features such as: 
using only scalable images, using images whose original size was larger than a threshold, or deciding if the 
user wants to know which images were discarded.

See a03_define_custom_dataset.py -h for more info about arguments.

For all datasets generated, it will show on screen some statistis of them, like number of images per class and DR level, and their csv files will 
be stored in root direcotry (Retinopathy/).

Use example: 
- (1) Create a dataset with 3 classes (class 0: dr levels 0 and 1, class 1: dr levels 2 and 3, class 2: dr level 4)
- (2) Using images from datasets eyepacs and aptos_2019
- (3) Divide it into train, validation and test in this way: 80 - 15 - 5
- (4) Ignore ungradable images
- (5) Ignore images whose original size was lower than 200 pixels
- (6) Save on a csv file all discarded images (perhaps for a future use)

> python ./a03_define_custom_dataset.py -d eyepacs aptos_2019 -c 01 23 4 --train_val_test 80 15 5 --only_gradable -s 200 --save_discarded 
                                        --------------------- ---------- ------------------------ --------------- ------ ----------------
                                                   2               1                 3                   4           5           6
'''

def show_dataset_statistics(dataset):
    # Create a copy of the dataset
    dataset = pd.DataFrame(dataset.sort_values(by=['label', 'DR_level'], ignore_index=True))

    # Get labels/classes and DR levels inside this dataset
    labels = dataset['label'].unique()
    dr_levels = dataset['DR_level'].unique()

    # Initialize statistics
    stats = {}
    for lb in labels:
        stats[lb] = {}
        # Add counter for each label / class
        stats[lb]['count'] = 0
        for dr in dr_levels:
            # Add counter for each DR level inside a class
            stats[lb][dr] = 0
    
    # Fill statistics
    for row in dataset.itertuples():
        # Increment counter for label / class
        stats[row.label]['count'] += 1
        # Increment counter for DR level inside this class
        stats[row.label][row.DR_level] += 1
    
    for lb in stats:
        # Get number of images belonging current class
        count = stats[lb]['count']
        # Remove this entry out of the dictionary
        del stats[lb]['count']

        # List with DR levels that have at least one image in current class
        dr_list = list(filter(lambda x: stats[lb][x] != 0, [lvl for lvl in stats[lb]]))

        # Print list of all DR levels present inside current class, showing the number of images and the percentage
        print('\tClass',lb,'has',count,'--',str(round((count/dataset.shape[0]) * 100, 2)) + '%','--','images distributed in these DR levels:\n\t\t'+\
              '\n\t\t'.join(['DR_lvl_' + str(lvl) + ' with ' + str(stats[lb][lvl]) + ' images -- ' +\
                                                    str(round((stats[lb][lvl] / count) * 100, 2)) + '%' for lvl in dr_list]))

# Define arguments
parser = argparse.ArgumentParser(description='Create CSV files for different DR classifications')

parser.add_argument('-d','--datasets',
                    nargs='+',
                    help='Specify which datasets will be used to create the custom dataset. These datasets must be stored in data/processed_datasets/',
                    default=['eyepacs'])

parser.add_argument('-c', '--classes', 
                    nargs='+', 
                    help='Give the DR Levels for each class. For example, to define 2 classes (no DR vs DR): 0 1234', 
                    default=['0', '1234']) # Add 'u' for specifying ungradable images as a class? - Pending

parser.add_argument('-s','--size', 
                    type=int, 
                    nargs=1, 
                    help='Specify the minimum size of original image (before cropping) to be selected', 
                    default=[0])

parser.add_argument('--train_val_test', 
                    type=int, 
                    nargs=3, 
                    help='Specify which proportion of the whole generated dataset will be used for train, validation and test. '\
                         'There must be 3 percentages, if a dataset is not needed (for example, create only train and validation datasets), '\
                         'it must be specified with 0. The sum of all percentages must not be greater than 100, and any value cannot be lower than 0',
                    default=[80, 15, 5])

parser.add_argument('--seed', 
                    type=int, 
                    nargs=1, 
                    help='Specify the seed when shuffling the new generated dataset', 
                    default=[42])

parser.add_argument('--name',
                    type=str,
                    nargs=1,
                    help='Specify the name of the CSV files to be generated')

parser.add_argument('--cardinality',
                    type=int,
                    nargs=1,
                    help='Specify the total number of images to be used',
                    default=[-1])

parser.add_argument('--only_gradable', 
                    action='store_true', # Returns true if this argument was given
                    help='Use only good quality (gradable) images', 
                    default=False)

parser.add_argument('--save_discarded', 
                    action='store_true', # Returns true if this argument was given
                    help='Save in a .csv file which images were excluded and their discarding reason', 
                    default=False)

args = parser.parse_args()

# Read arguments ------------------------------------
datasets_names = args.datasets

# classes = []
# for pos, i in enumerate(args.classes):
#     classes.append([])
#     for j in i:
#         classes[pos].append(int(j))

# Convert classes arguments to list of lists of integers
classes = [[int(j) for j in i] for i in args.classes]

minimum_size = args.size[0]
only_gradable = args.only_gradable
save_discarded = args.save_discarded

# Get train, validation and test percentages
percentages = args.train_val_test

# Get total number of images to extract from specified datasets
total_num_images = args.cardinality[0]

# Check that all datasets given exists
for dataset_name in datasets_names:
    if not os.path.exists('data/processed_datasets/0_csvs/' + dataset_name + '.csv'):
        print('Dataset \'' + dataset_name + '\' does not exist. Please check that this dataset has been processed and stored in data/processed_datasets/ ' + \
              'using a02_redistribute_datasets.py script')
        sys.exit(0)

# Check that percentages are correct
if np.min(np.array(percentages)) < 0:
    print('All percentages must be equals or greater than 0')
    sys.exit(0)

if np.sum(np.array(percentages)) > 100:
    print('Sum of all percentages for train, validation and test datasets must not exceed 100')
    sys.exit(0)

# Get seed for shuffling
seed = args.seed[0]

# Get name for csvs
if args.name is not None:
    name_csv = args.name[0]
else:
    name_csv = None

print('-------------------')

# Load datasets
datasets = list(map(lambda x : pd.read_csv('data/processed_datasets/0_csvs/' + x + '.csv'), datasets_names))

# Check that there is at least one image (in any of the given datasets) that belongs to any of the DR levels specified for each class defined 
# because some datasets may not have all 5 DR levels
# For example: when trying to define 01, 23, 4 classes using messidor dataset, which only have DR labels from 0 to 3,
# it cannot create a class 3 (class defined with only DR level 4 images) becase there is no image provided with that DR level
ocurrences = {}
for n, d in zip(datasets_names, datasets):
    # Get all DR levels present in current dataset
    # Convert then to int to easier use
    lbls = list(map(int, list(d['DR_level'].unique())))
    print('Dataset',n,'have these DR levels:',np.sort(lbls))
    # Check if there are unknown DR level images, they will not be used
    if -1 in lbls:
        lbls.remove(-1)
        print('-- There are unknown DR level images (-1). These images will be discarded')
    for lb in lbls:
        # Mark as true if a DR level is present in a dataset
        if ocurrences.get(lb) is None:
            ocurrences[lb] = True

# Once all DR levels of all datasets specified have been counted, check if every new class has images (reason explained before)
for pos_in_arguments, class_ in enumerate(classes):
    ok = False
    # Check if at least one of the DR levels belonging to the current class is present in any of the specified datasets
    for subclass in class_:
        if ocurrences[subclass] is not None:
            ok = True
            break
    if not ok:
        print('None of the datasets given has labeled images for any of these DR levels',class_,'which is the class specified in',(pos_in_arguments+1),'position in the arguments')
        sys.exit(0)

# Another check. If the option '--only_gradable' was present but any of the specified datasets does not have gradability labels, print a message and continue
if only_gradable:
    for n, d in zip(datasets_names, datasets):
        # Get gradability labels and cast them to int for easier use
        grads = list(map(int, list(d['gradability'].unique())))
        # If there is only one label (-1), no image has gradablity label
        if len(grads) == 1 and grads[0] == -1:
            print('** NOTICE: Although \'--only_gradable\' argument was specified, dataset \'' + n + '\' does not have gradability labels. All images will be considered as gradable')

# Create the csv file -------------------------

# The csv file will have 3 columns: path, label, original DR level
# Create empty lists for these fields
paths = []
labels = []
dr_level = []

# Create empty lists for discarded images and their reason
# If 'save_discarded' option is present, a csv file will be generated with these lists
discarded = []
reason = []

def find_class(dr_level):
    '''
    Get in which new class does the DR level given belong 
    '''
    for i, c in enumerate(classes):
        if dr_level in c:
            return i
    # Return unnecessary, because it was already checked that for each class there was at least one dataset with one of its DR levels
    return 0

# For each dataset, get every image and check if it has to be discarded or not
for dataset in datasets:
    for row in dataset.itertuples():
        # Discard image if image is ungradable and only gradable images are required
        if only_gradable and row.gradability == 0:
            # Save path of discarded image and its reason
            discarded.append(row.path)
            reason.append('ungradable')

        # Discard image if its original size was not large enough
        elif minimum_size > row.old_size_x or minimum_size > row.old_size_y:
            # Save path of discarded image and its reason
            discarded.append(row.path)
            reason.append('insufficient_original_size')
        else:
            # Check if image has DR level, discard if not
            if row.DR_level != -1:
                paths.append(row.path)
                labels.append(find_class(row.DR_level)) # Set as label the new class
                dr_level.append(row.DR_level)
            else:
                # Save path of discarded image and its reason
                discarded.append(row.path)
                reason.append('unknown_DR_level')

print('-------------------')
if len(discarded) != 0:
    print('Discarded',len(discarded),'images')
else:
    print('No image was discarded')
print('-------------------')

# Create new DataFrame with all non-discarded images
csv_file = pd.DataFrame({
    'path' : paths,
    'label' : labels,
    'DR_level' : dr_level # This field is useful
})

# Shuffle all data, using the specified seed
csv_file = csv_file.sample(frac=1,random_state=seed)

# Get the number of images specified
if total_num_images > 0:
    if csv_file.shape[0] < total_num_images:
        print('There are not enough images to reach the specified total cardinality:',total_num_images)
        print('The total number of images will be:', csv_file.shape[0])
    else:
        # Get only the specified number of images
        print('There are', csv_file.shape[0], 'images. Only', total_num_images, 'will be used')
        csv_file = csv_file.head(total_num_images)

# Save all data in a .csv file - If no name was given, create one with all arguments given
if name_csv == None:
    name_csv = '-'.join(datasets_names) + '-' + \
            'classes-' + '-'.join([''.join(list(map(str,class_))) for class_ in classes]) + \
            ('-grad' if only_gradable else '') + \
            (('-sz_' + str(minimum_size)) if minimum_size > 0 else '') + \
            (('-numImgs_' + str(total_num_images)) if total_num_images > 0 else '')

# Always replace the csv file if another file with same name existed previously
if os.path.exists(name_csv + '-ALL.csv'):
    os.remove(name_csv + '-ALL.csv')

# Save csv file
csv_file.to_csv(name_csv + '-ALL.csv', index=False)

print('Generated',name_csv + '-ALL.csv','file with all',csv_file.shape[0],'images')
print('-------------------')

# Split this dataset in train, validation and test datasets, using specified percentages

# Names for these datasets
split_datasets = ['TRAIN', 'VALIDATION', 'TEST']

# Convert percentages to [0, 1] values --> Easier to multiply
percentages_f = list(map(lambda x: x / 100.0, percentages))

# Actual number of saved images
used_images = 0

total_num_images = csv_file.shape[0]

for i, percentage in enumerate(percentages_f):
    if percentage != 0:
        # Get the current percentage from all images
        current_data = csv_file.iloc[used_images : int(used_images + np.round(percentage * total_num_images))]

        # File name: name + dataset + percentage + .csv
        file_name = name_csv + '-' + split_datasets[i] + '-' + str(percentages[i]).zfill(2) + '.csv'

        # Always replace the csv file if another file with same name existed previously
        if os.path.exists(file_name):
            os.remove(file_name)
        
        # Save these images to a new csv
        current_data.to_csv(file_name, index=False)

        # Show information about dataset
        print(' - Generated',file_name,'containing',current_data.shape[0],'images belonging to these classes:',', '.join(list(map(str, np.sort(current_data['label'].unique())))))
        # Print some statistics
        show_dataset_statistics(current_data)

        # Increase used images counter
        used_images += int(np.round(percentage * total_num_images))
    else:
        print('No',split_datasets[i].lower(),'dataset has been created')

# If 'save_discarded' option was present in arguments, save those discarded images and their discarding reason in a csv file
if save_discarded:
    print('-------------------')
    # Create file if there are discarded images
    if len(discarded) != 0:
        # Structure for discarded csv file: path to image and discarding reason
        csv_discarded = pd.DataFrame({
            'path' : discarded, 
            'reason' : reason
        })

        name = 'discarded.csv'

        # Always replace the csv file if another file with same name existed previously
        if os.path.exists(name):
            os.remove(name)

        # Save csv file
        csv_discarded.to_csv(name, index=False)

        print('Generated',name,'file with all discarded images and their discarding reason')
    else:
        print('No image was discarded')