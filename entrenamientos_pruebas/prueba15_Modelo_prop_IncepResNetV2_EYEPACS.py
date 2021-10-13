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
VALIDATION_FILE = 'VALIDATION_BALANCED_0_1_234.csv'

ONE_HOT_FORMAT = False

TRAINING_BATCH_SIZE = 12
TRAINING_DATA_AUG = True
TRAINING_BALANCED = False
TRAINING_PREFETCH = 2
TRAINING_TAKE_SIZE = None

VALIDATION_BATCH_SIZE = 12
VALIDATION_DATA_AUG = False
VALIDATION_BALANCED = False
VALIDATION_PREFETCH = 2
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
# Compute class weights

class_weights = compute_class_weight('balanced', classes=np.unique(y_true_train[:,0]), y=y_true_train[:,0])
d_class_weights = dict(enumerate(class_weights))
print(d_class_weights)

########################################################################################
# Define model
model = models.resNet50V2.get_model(input_shape=(540,540,3), num_outputs=len(DR_LEVELS_PER_CLASS))
model.summary()

########################################################################################
# Defining saving paths

base_path = 'saved_weights/resNet50V2/RMSProp/'

if not os.path.exists(base_path):
    os.mkdir(base_path)

# Save ground truth for validation dataset
np.save(base_path + 'ground_truth_val.npy', y_true_val)

########################################################################################
# Define metrics
classes_names = [''.join(list(map(str,class_))) for class_ in DR_LEVELS_PER_CLASS]
print(classes_names)

classes_for_metric = [1, 2]

metric_AUC_0_1234 = metrics.Wrapper_AUC(classes=classes_for_metric, original_dr_lvls=classes_names, is_one_hot=ONE_HOT_FORMAT)

metric_Sp_at_95_Sens_0_1234 = metrics.Wrapper_SpecificityAtSensitivity(sensitivity=0.95, classes=classes_for_metric, original_dr_lvls=classes_names, is_one_hot=ONE_HOT_FORMAT)

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
          tf.keras.callbacks.LearningRateScheduler(callbacks.create_scheduler_function(10, 0.9))
]

########################################################################################
# Compile and train model

validation_dir = base_path+'validation_outputs/'

if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

'''
From Voets' GitHub (Google replication 2018-19)
# This is TF 1.X
decay = 4e-5 # This value is the weight decay specified in their papers
...
train_op = tf.train.RMSPropOptimizer(
    learning_rate=learning_rate, decay=decay) \ # so, in TF 1.x, RMSProp has an argument called 'decay' for this purpose
        .minimize(loss=mean_xentropy, global_step=global_step)

### TF 1.x RMSProp description
https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/RMSPropOptimizer

Argument 'decay' -- Discounting factor for the history/coming gradient

### TF 2.x RMSProp description
https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop

Argument 'rho' -- Discounting factor for the history/coming gradient. Defaults to 0.9.

So, 'rho' should be the weight decay argument specified on Voets' code
'''

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=4e-5, clipnorm=1.0), # There is no momentum reference in the papers
              metrics=['accuracy',
                       metric_AUC_0_1234, # DR levels: 0 vs 1,2,3,4
                       metric_Sp_at_95_Sens_0_1234, # DR levels: 0 vs 1,2,3,4
                       metrics.RunningValidation(path=validation_dir, n_columns=len(DR_LEVELS_PER_CLASS)) # THIS METRIC MUST BE INIZIALIZATED BEFORE EVERY TRAINING
                      ]) 

num_epochs = 2000

# history = model.fit(train_dataset.take(50), epochs=num_epochs, validation_data=val_dataset.take(10), verbose=1, callbacks=cbacks, class_weight=d_class_weights)
history = model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, verbose=1, callbacks=cbacks, class_weight=d_class_weights)
