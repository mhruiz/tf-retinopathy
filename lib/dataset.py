from imgaug import parameters as iap
import imgaug.augmenters as iaa
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from . import data_augmentation as data_aug

#--------------------------------------------------------------------------------------------------------
'''
Mapping functions
'''

# Mapping function for reading image paths
def read_images(img, label):
    img_raw = tf.io.read_file(img)
    return tf.io.decode_png(img_raw, channels=3), label

# Mapping function on tf.data.Dataset
def tf_apply_data_augmentation(image, label):
    
    # Py function that must be wrapped inside tf.py_function()
    # If will apply all data augmentation defined
    def fn(image):
        return data_aug.py_apply_data_augmentation(image=image.numpy())

    # Get image dtype
    im_dtype = image.dtype
    
    # It's importat to get the image shape before process it -- TensorFlow website
    im_shape = image.shape 
    
    # tf.py_function wraps a python function into a TensorFlow op that executes it eagerly
    [image_aug, ] = tf.py_function(fn, [image], [im_dtype])
    
    # Recover shape -- TensorFlow website
    image_aug.set_shape(im_shape) 

    return image_aug, label


def tf_apply_data_augmentation_2(image, label):
    # Py function that must be wrapped inside tf.py_function()
    # If will apply all data augmentation defined
    def fn(image):
        return data_aug.py_apply_data_augmentation_2(image=image.numpy())

    # Get image dtype
    im_dtype = image.dtype
    
    # It's importat to get the image shape before process it -- TensorFlow website
    im_shape = image.shape 
    
    # tf.py_function wraps a python function into a TensorFlow op that executes it eagerly
    [image_aug, ] = tf.py_function(fn, [image], [im_dtype])
    
    # Recover shape -- TensorFlow website
    image_aug.set_shape(im_shape) 

    return image_aug, label


def normalize_images(img, label):
    img = tf.cast(img, dtype=tf.float32)
    img = img / 255.
    return img, label

from time import time

#--------------------------------------------------------------------------------------------------------
'''
Create dataset function

def create_dataset(csv_file, 
                   num_classes,
                   balanced=False,
                   apply_data_augmentation=False,
                   batch_size=1, 
                   prefetch_buffer=None, 
                   repeat=1,
                   shuffle=False,
                   size=None):

    # Read file
    dataset = pd.read_csv(csv_file)
    
    images = dataset['path'].tolist()
    labels = dataset['label'].tolist()
    dr_levels = dataset['DR_level'].tolist()

    if size is not None:
        if type(size) is float:
            take = round(len(images)*size)
        else:
            take = size

        images = images[:take]
        labels = labels[:take]
        dr_levels = dr_levels[:take]

        print(take, 'images are originally taken from csv file')

    # Get true labels as numpy array
    if num_classes == 2:
        true_labels = np.column_stack((np.array(labels, dtype=np.uint8), 
                                    np.array(dr_levels, dtype=np.uint8)))
    else:
        true_labels = np.column_stack((np.array(tf.keras.utils.to_categorical(labels, num_classes=num_classes), dtype=np.uint8), 
                                    np.array(dr_levels, dtype=np.uint8)))

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Convert to tensorflow.data.Dataset

    if balanced:

        # Define a dictionary with an entry for each class. As value, it will have a tuple with 2 lists: paths and labels or classes
        separated_data = {i:([],[]) for i in range(num_classes)}

        for i, lb in enumerate(labels):
            # Image path
            separated_data[lb][0].append(images[i])
            # Label
            separated_data[lb][1].append(lb)
        
        # Create a list with all datasets but loaded as tf.data.Dataset objects
        separated_datasets = []

        # Find which is the size or cardinality of the smallest class
        smallest_size = np.Inf

        for lb in separated_data:

            # Create binary dataset
            if num_classes == 2:
                separated_datasets.append(tf.data.Dataset.from_tensor_slices((separated_data[lb][0], 
                                                                              separated_data[lb][1])))
            # Create categorical dataset
            elif num_classes > 2:
                separated_datasets.append(tf.data.Dataset.from_tensor_slices((separated_data[lb][0], 
                                                                              tf.keras.utils.to_categorical(separated_data[lb][1], 
                                                                                                            num_classes=num_classes))))
            # Check if current dataset is the smallest
            if len(separated_data[lb][0]) < smallest_size:
                smallest_size = len(separated_data[lb][0])

        total_num = 0
        # For each dataset, take the same number of elements. Apply first shuffle if needed
        for i, dt in enumerate(separated_datasets):
            if shuffle:
                dt = dt.shuffle(tf.data.experimental.cardinality(dt).numpy())

            # Get from every dataset, the same number of examples
            dt = dt.take(smallest_size)

            total_num += smallest_size

            separated_datasets[i] = dt

        print('This balanced dataset will have', total_num, 'images, having', total_num//len(separated_datasets), 'images per class')
        
        # Concatenate all these datasets
        dataset = separated_datasets[0]
        for i in range(1, len(separated_datasets)):
            dataset = dataset.concatenate(separated_datasets[i])

    else:
        # If no class-balance is needed, just create tf.data.Dataset objects
        # Binary
        if num_classes == 2:
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        # Categorical
        elif num_classes > 2:
            dataset = tf.data.Dataset.from_tensor_slices((images, tf.keras.utils.to_categorical(labels, num_classes=num_classes)))

    # Read images
    dataset = dataset.map(read_images, num_parallel_calls=AUTOTUNE)

    if apply_data_augmentation:
        dataset = dataset.map(tf_apply_data_augmentation, num_parallel_calls=AUTOTUNE)

    # Normalize images
    dataset = dataset.map(normalize_images, num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset).numpy() // 10)

    dataset = dataset.batch(batch_size)

    # dataset = dataset.repeat(repeat)

    if prefetch_buffer is not None:
        dataset = dataset.prefetch(prefetch_buffer)

    return dataset, true_labels
'''

def create_dataset_new(csv_file, 
                        list_list_classes,
                        balanced=False,
                        apply_data_augmentation=False,
                        use_new_augmenter=False,
                        batch_size=1, 
                        prefetch_buffer=None, 
                        is_validation=False,
                        one_hot_format=True,
                        size=None, 
                        val_data_aug=False,
                        check_exists_all_dr_lvls=True):

    # Se aplica shuffle por defecto siempre

    # Read csv file
    dataset = pd.read_csv(csv_file)
    
    image_paths = dataset['path'].tolist()
    dr_levels = dataset['DR_level'].tolist()

    # If 'size' is given, take the first images from csv
    if size is not None:
        print('Size no es none')
        if type(size) is float:
            # If size is float, it means the percentage of images that has to be taken
            take = round(len(image_paths)*size)
        else:
            # It's the specific number of images
            take = size

        image_paths = image_paths[:take]
        dr_levels = dr_levels[:take]

        print(take, 'image_paths are originally taken from csv file')

    try:
        # Check that all Dr levels specified in 'list_list_classes' are in the dataset
        assert all(dr_lvl in set(dr_levels) for dr_lvl in [dr for class_ in list_list_classes for dr in class_]), \
            'There is at least one DR level specified in \'list_list_classes\' that is not present in selected images.\n' + \
                'Specified DR levels: ' + ', '.join([str(dr) for class_ in list_list_classes for dr in class_]) + '\n' + \
                    'Existing DR levels: ' + ', '.join(list(map(str,list(set(dr_levels)))))
    except AssertionError as e:
        if check_exists_all_dr_lvls:
            raise e
        else:
            print(e)
            print('Execution continues...')

    # Check if 'list_list_classes' is sorted: from lower to higher Drlvls
    previous_max = max(list_list_classes[0])
    used_dr_lvls = set()
    for i in range(1, len(list_list_classes)):

        # Also check that there is no repetitions
        if any(dr_lvl in used_dr_lvls for dr_lvl in list_list_classes[i]):
            raise Exception('There is at least one DR label repeated in two different classes')

        current_max = max(list_list_classes[i])
        assert current_max > previous_max, 'DR levels are not sorted. Every class must have higher DR levels than previous classes'

        # Add current Dr levels to 'used_dr_lvls' for checking repetitions on further classes
        used_dr_lvls.update(list_list_classes[i])

        previous_max = current_max

    # Build labels array
    # As example, list_list_classes should have this form: [[0], [1], [2,3,4]]
    asignations = [-1 for i in list_list_classes for j in i] # [[0], [1], [2,3,4]] --> [-1, -1, -1, -1, -1]

    for i, class_ in enumerate(list_list_classes):
        for dr_lvl in class_:
            asignations[dr_lvl] = i

    labels = [asignations[dr_lvl] for dr_lvl in dr_levels]

    num_classes = max(labels) + 1

    # Get true labels as numpy array
    if one_hot_format:
        true_labels = np.column_stack((np.array(tf.keras.utils.to_categorical(labels, num_classes=num_classes), dtype=np.uint8), 
                                    np.array(dr_levels, dtype=np.uint8)))
    else:
        true_labels = np.column_stack((np.array(labels, dtype=np.uint8), np.array(dr_levels, dtype=np.uint8)))

    # Convert to tensorflow.data.Dataset object

    if balanced:

        # Define a dictionary with an entry for each class. As value, it will have a tuple with 2 lists: paths and labels or classes
        separated_data = {i:([],[]) for i in range(num_classes)}

        for i, lb in enumerate(labels):
            # Image path
            separated_data[lb][0].append(image_paths[i])
            # Label
            separated_data[lb][1].append(lb)
        
        # Create a list with all datasets but loaded as tf.data.Dataset objects
        separated_datasets = []

        # Find which is the size or cardinality of the smallest class
        smallest_size = np.Inf

        print('Checking num images per label...')

        for lb in separated_data:
            if one_hot_format:
                separated_datasets.append(tf.data.Dataset.from_tensor_slices((separated_data[lb][0], 
                                                                            tf.keras.utils.to_categorical(separated_data[lb][1], 
                                                                                                        num_classes=num_classes))))
            else:
                separated_datasets.append(tf.data.Dataset.from_tensor_slices((separated_data[lb][0], separated_data[lb][1])))

            print('Class/label', lb, 'has', len(separated_data[lb][0]), 'images')

            # Check if current dataset is the smallest
            if len(separated_data[lb][0]) < smallest_size:
                smallest_size = len(separated_data[lb][0])

        total_num = 0
        # For each dataset, take the same number of elements. Apply first shuffle if needed
        for i, dt in enumerate(separated_datasets):
            # If shuffle is False, it will always take the same elements per class
            if not is_validation:
                print('Apply shuffle on each one-label dataset')
                dt = dt.shuffle(tf.data.experimental.cardinality(dt).numpy())

            # Take the same number of examples for each class
            dt = dt.take(smallest_size)

            total_num += smallest_size

            separated_datasets[i] = dt

        print('This balanced dataset will have', total_num, 'images, having', smallest_size, 'images per class')
        
        # Concatenate all these datasets
        dataset = separated_datasets[0]
        for i in range(1, len(separated_datasets)):
            dataset = dataset.concatenate(separated_datasets[i])
        
        print('One-label datasets joined into a one single tf.data.Dataset object')

    else:
        # If no class-balance is needed, just create tf.data.Dataset objects
        if one_hot_format:
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, tf.keras.utils.to_categorical(labels, num_classes=num_classes)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle before any mapping function
    if not is_validation:
        print('Shuffle all dataset')
        dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset).numpy())

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Read images
    dataset = dataset.map(read_images, num_parallel_calls=AUTOTUNE)

    if apply_data_augmentation:
        print('Se aplica data aug')
        if use_new_augmenter:
            dataset = dataset.map(tf_apply_data_augmentation_2, num_parallel_calls=AUTOTUNE)
        else:
            dataset = dataset.map(tf_apply_data_augmentation, num_parallel_calls=AUTOTUNE)

    if is_validation and val_data_aug:
        print('Se aplica data aug para val -- NO IMPLEMENTADO')

    # Normalize images
    dataset = dataset.map(normalize_images, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)

    if prefetch_buffer is not None:
        dataset = dataset.prefetch(prefetch_buffer)

    # Return the generated dataset and its ground truth outputs if shuffle was False
    # If shuffle is True, 'true_labels' order will not match with 'dataset' order
    return dataset, true_labels # true_labels are necessary for obtaining class weights