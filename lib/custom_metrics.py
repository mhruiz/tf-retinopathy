import tensorflow as tf
import numpy as np

'''
These classes are wrappers for Keras' metrics which were designed for binary (+/-) classification.
All these classes will have a 'classes' argument which will be a list of integer, specifying which columns have to be joined (by addition)
in order to get a single column like a binary classification.
'''

class Wrapper_Base(tf.keras.metrics.Metric):

    def __init__(self, metric, classes, name, dtype, original_dr_lvls=['0','1','234'], is_one_hot=True):

        all_classes = list(range(len(original_dr_lvls)))
        for e in classes:
            all_classes.remove(e)
        
        # Negative (Dr levels that does not belong to the positive class) - positive
        class_names = ''.join([original_dr_lvls[x] for x in all_classes]) + '_' + ''.join([original_dr_lvls[x] for x in classes])

        super().__init__(name=name + '_DRlvls_' + class_names, dtype=dtype)

        self.num_total_classes = len(original_dr_lvls)

        self.is_one_hot = is_one_hot

        self.custom_name = name + '_DRlvls_' + class_names

        self.classes = classes
        self.metric = metric

    def get_custom_name(self):
        return self.custom_name

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Convert a multi-class prediction into a binary prediction
        # We only need to check the specified columns, joined into a single one

        # If dataset was not loaded into a one-hot format, it's necessary to convert ground truth array into one-hot
        if not self.is_one_hot:
            y_true = tf.one_hot(y_true, self.num_total_classes)[:,0,:]

        # tf.gather with axis=1 -- get specified columns
        # tf.reduce_sum with axis=1 -- add columns, returning a (None,) shape Tensor
        y_true_joined = tf.reduce_sum(tf.gather(y_true, indices=self.classes, axis=1), axis=1)
        y_pred_joined = tf.reduce_sum(tf.gather(y_pred, indices=self.classes, axis=1), axis=1)

        # Sometimes the addition of prediction columns could result in numbers higher than 1, so we apply clip_by_value
        self.metric.update_state(y_true_joined, tf.clip_by_value(y_pred_joined, 0, 1)) 

    def result(self):
        return self.metric.result()

    def reset_states(self):
        return self.metric.reset_states()

class Wrapper_SpecificityAtSensitivity(Wrapper_Base):
    def __init__(self, sensitivity, classes, original_dr_lvls, is_one_hot):
        super().__init__(metric=tf.keras.metrics.SpecificityAtSensitivity(sensitivity),
                         name='Sp_at_' + str(round(100 * sensitivity)) + '_sens', 
                         classes=classes,
                         dtype=tf.float32,
                         original_dr_lvls=original_dr_lvls,
                         is_one_hot=is_one_hot)

class Wrapper_SensitivityAtSpecificity(Wrapper_Base):
    def __init__(self, specificity, classes, original_dr_lvls, is_one_hot):
        super().__init__(metric=tf.keras.metrics.SensitivityAtSpecificity(specificity),
                         name='Sens_at_' + str(round(100 * specificity)) + '_sp', 
                         classes=classes,
                         dtype=tf.float32,
                         original_dr_lvls=original_dr_lvls,
                         is_one_hot=is_one_hot)

class Wrapper_AUC(Wrapper_Base):
    def __init__(self, classes, original_dr_lvls, is_one_hot):
        super().__init__(metric=tf.keras.metrics.AUC(summation_method='minoring'), 
                                        # 'minoring' applies left summation for increasing intervals and right summation for decreasing intervals
                         name='AUC', 
                         classes=classes,
                         dtype=tf.float32,
                         original_dr_lvls=original_dr_lvls,
                         is_one_hot=is_one_hot)

'''
This 'metric' will save every prediction obtanined for each validation sample,
supposing that there will be always a two stage epochs: train + val.

It will be useful for recovering validation's predictions on past epochs.
'''
class RunningValidation(tf.keras.metrics.Metric):

    def __init__(self, path, n_columns=2):
        super().__init__(name="RunningValidation")

        # Boolean variable to check if validation is running, initialized to False (1st training, 2nd validation)
        self.is_validation_step = tf.Variable(False)

        self.n_columns = n_columns

        # Variable where all predictions (on validation dataset) will be saved
        # Initialize with a zeros row
        self.model_output = tf.Variable([[0.] * n_columns], shape=(None,n_columns), validate_shape=False)

        # Epoch counter
        self.epoch = tf.Variable(0, dtype=tf.int32)

        # Save path
        self.path = path

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.is_validation_step.value():
            # Concat previous data with current validation batch
            self.model_output.assign(tf.concat([self.model_output.value(), y_pred], axis=0))

    def result(self):
        if self.is_validation_step.value():
            return 1.
        else:
            return 0.

    def reset_states(self):

        if self.is_validation_step:
            # Save model prediction on validation dataset after every epoch as a binary numpy file. This kind of files take up very little disk space
            np.save(self.path + 'output_epoch_' + str(int(self.epoch.value())).zfill(4) + '.npy', self.model_output.value().numpy()[1:,:])

            self.epoch.assign_add(1)

        # Change between training and validation stage
        self.is_validation_step.assign(not self.is_validation_step.value())

        # Reset validation predictions
        self.model_output.assign([[0.] * self.n_columns])

    def reset_all(self):
        self.is_validation_step.assign(False)
        self.model_output.assign([[0.] * self.n_columns])
        self.epoch.assign(0)
