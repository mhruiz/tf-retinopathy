from sklearn.utils.class_weight import compute_class_weight

import lib.custom_callbacks as callbacks
import lib.custom_metrics as metrics
import lib.evaluation as ev
import lib.plotting as plot
import lib.models as models
import lib.dataset as dt

import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import sys
import os


# Initialize GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

########################################################################################
# ARGUMENTS

DR_LEVELS_PER_CLASS = [[0], [1], [2,3,4]]

IMAGE_SIZE = (540, 540, 3)

# Specify dataset files
TRAIN_FILE = 'DATASET-TRAIN-80.csv'
VALIDATION_FILE = 'DATASET-VALIDATION-10-BALANCED-0-1-234.csv'

ONE_HOT_FORMAT = False

TRAINING_BATCH_SIZE = 4
TRAINING_DATA_AUG = True
TRAINING_BALANCED = True
TRAINING_PREFETCH = 20
TRAINING_TAKE_SIZE = None

VALIDATION_BATCH_SIZE = 12
VALIDATION_DATA_AUG = False
VALIDATION_BALANCED = False
VALIDATION_PREFETCH = 1
VALIDATION_TAKE_SIZE = None

########################################################################################
# Create datasets

train_dataset, y_true_train = dt.create_dataset_new(TRAIN_FILE, 
                                                    DR_LEVELS_PER_CLASS, 
                                                    balanced=TRAINING_BALANCED, 
                                                    apply_data_augmentation=TRAINING_DATA_AUG, 
                                                    batch_size=TRAINING_BATCH_SIZE, 
                                                    prefetch_buffer=TRAINING_PREFETCH, 
                                                    one_hot_format=ONE_HOT_FORMAT,
                                                    size=TRAINING_TAKE_SIZE)

val_dataset, y_true_val = dt.create_dataset_new(VALIDATION_FILE, 
                                                DR_LEVELS_PER_CLASS, 
                                                balanced=VALIDATION_BALANCED,
                                                apply_data_augmentation=VALIDATION_DATA_AUG,
                                                batch_size=VALIDATION_BATCH_SIZE,
                                                prefetch_buffer=VALIDATION_PREFETCH, 
                                                is_validation=True,
                                                one_hot_format=ONE_HOT_FORMAT,
                                                size=VALIDATION_TAKE_SIZE)

########################################################################################
# Define model
model = models.efficientNetB5.get_model(input_shape=(540,540,3), num_outputs=len(DR_LEVELS_PER_CLASS))
model.summary()

########################################################################################
# Defining saving paths

base_path = 'saved_weights/efficientNetB5/SGD_bal_0-1-234_bs4_NoLRSCH/'
base_path_splitted = base_path.split('/')
for i in range(len(base_path_splitted)):
    path_i = '/'.join(base_path_splitted[:i+1])
    if not os.path.exists(path_i):
        os.mkdir(path_i)

# Save ground truth for validation dataset
np.save(base_path + 'ground_truth_val.npy', y_true_val)

########################################################################################
# Define metrics
classes_names = [''.join(list(map(str,class_))) for class_ in DR_LEVELS_PER_CLASS]
print(classes_names)

classes_for_metric = [1, 2]
metric_AUC_0_1234 = metrics.Wrapper_AUC(classes=classes_for_metric, original_dr_lvls=classes_names, is_one_hot=ONE_HOT_FORMAT)
metric_Sp_at_95_Sens_0_1234 = metrics.Wrapper_SpecificityAtSensitivity(sensitivity=0.95, classes=classes_for_metric, original_dr_lvls=classes_names, is_one_hot=ONE_HOT_FORMAT)

classes_for_metric = [2]
metric_AUC_01_234 = metrics.Wrapper_AUC(classes=classes_for_metric, original_dr_lvls=classes_names, is_one_hot=ONE_HOT_FORMAT)
metric_Sp_at_95_Sens_01_234 = metrics.Wrapper_SpecificityAtSensitivity(sensitivity=0.95, classes=classes_for_metric, original_dr_lvls=classes_names, is_one_hot=ONE_HOT_FORMAT)

########################################################################################
# Define callbacks
cbacks = [tf.keras.callbacks.ModelCheckpoint(base_path + 'best_' + metric_AUC_0_1234.get_custom_name() + '.h5', 
                                             monitor='val_' + metric_AUC_0_1234.get_custom_name(),
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='max'),

          tf.keras.callbacks.ModelCheckpoint(base_path + 'best_' + metric_Sp_at_95_Sens_0_1234.get_custom_name() + '.h5', 
                                             monitor='val_' + metric_Sp_at_95_Sens_0_1234.get_custom_name(),
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='max'),
          
          tf.keras.callbacks.ModelCheckpoint(base_path + 'best_' + metric_AUC_01_234.get_custom_name() + '.h5', 
                                             monitor='val_' + metric_AUC_01_234.get_custom_name(),
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='max'),

          tf.keras.callbacks.ModelCheckpoint(base_path + 'best_' + metric_Sp_at_95_Sens_01_234.get_custom_name() + '.h5', 
                                             monitor='val_' + metric_Sp_at_95_Sens_01_234.get_custom_name(),
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='max'),

          tf.keras.callbacks.ModelCheckpoint(base_path + 'best_accuracy.h5',
                                             monitor='val_accuracy',
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='max'),

          tf.keras.callbacks.ModelCheckpoint(base_path + 'best_loss.h5',
                                             monitor='val_loss',
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='min'),
          
          callbacks.Save_Training_Evolution(base_path + 'training_evolution.csv'),
          
          # learning rate scheduler
          # tf.keras.callbacks.LearningRateScheduler(callbacks.create_scheduler_function(10, 0.9))
]

########################################################################################
# Compile and train model
validation_dir = base_path+'validation_outputs/'

if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, clipnorm=1.0), 
              metrics=['accuracy',
                       metric_AUC_0_1234, # DR levels: 0 vs 1,2,3,4
                       metric_Sp_at_95_Sens_0_1234, # DR levels: 0 vs 1,2,3,4
                       metric_AUC_01_234, # DR levels: 0,1 vs 2,3,4
                       metric_Sp_at_95_Sens_01_234, # DR levels: 0,1 vs 2,3,4
                       metrics.RunningValidation(path=validation_dir, n_columns=len(DR_LEVELS_PER_CLASS)) # THIS METRIC MUST BE INIZIALIZATED BEFORE EVERY TRAINING
                      ]) 

num_epochs = 2000

history = model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, verbose=2, callbacks=cbacks)
