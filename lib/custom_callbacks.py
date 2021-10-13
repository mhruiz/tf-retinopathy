import tensorflow as tf
import os

class Save_Training_Evolution(tf.keras.callbacks.Callback):

    def __init__(self, filename):
        super().__init__()
        
        self.first_call = True
        self.filename = filename
    
    def on_epoch_end(self, epoch, logs=None):
        # On first call (end of first epoch), create a new text file, with headers (one per metric)
        # This file will have a csv file structure (comma separation)
        if self.first_call:
            self.first_call = False
            if os.path.exists(self.filename):
                os.remove(self.filename)
            with open(self.filename, 'a') as f:
                f.write(','.join(k for k in logs) + '\n')
        # Write on last line current metrics
        with open(self.filename, 'a') as f:
            f.write(','.join(list(map(str, [logs[k] for k in logs]))) + '\n')

'''
Function utilized to decrease learning rate by a factor
'''
def lr_modifier(epoch, lr, num_steps, factor=0.9):
    if epoch != 0 and epoch % num_steps == 0:
        new_lr = lr * factor
        print('Epoch', str(epoch).zfill(4),'Learning rate reduced to:', new_lr)
        return new_lr
    return lr

def create_scheduler_function(num_steps, factor=0.9):

    def scheduler(epoch, learning_rate):
        return lr_modifier(epoch, learning_rate, num_steps, factor)

    return scheduler