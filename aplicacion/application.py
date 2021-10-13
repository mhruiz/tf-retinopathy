import sys

import tensorflow as tf
import numpy as np
import cv2

import preprocess

# Initialize GPU if exists

# GPU memory limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# These thresholds provided 90% sens on validation dataset
# 0 vs 1 vs 234
# acc_thresholds = [(0.10821643286573146,), (0.09018036072144288,), (0.2344689378757515,)]

# DR_LEVELS_PER_CLASS = [[0], [1], [2,3,4]]

# 0 vs 1234

with open('APPDATA/thresholds.dat', 'r') as f:
    
    # Only non empty text lines without '#' will remain
    remove_comments = lambda x: x.split('#')[0].strip()
    
    thresholds = filter(remove_comments, f.readlines())
    
# Convert thresholds to float and get the first one
threshold = list(map(float, thresholds))[0]

# Prepare for use
acc_thresholds = [(threshold,)]

DR_LEVELS_PER_CLASS = [[0], [1,2,3,4]]

model_config = 'APPDATA/model_config_0vs1234.json'
model_weights = 'APPDATA/model_weights_0vs1234.h5'

image_size = (540, 540)


def load_model(path_structure, path_weights):

    with open(path_structure, 'r') as json_file:
        model = tf.keras.models.model_from_json(json_file.read())

    model.load_weights(path_weights)

    return model


def get_predicted_labels(pred, thresholds):

    if pred.shape[1] > 2:
        pred_lb = np.zeros((pred.shape[0],))

        for i in range(pred.shape[0]):

            is_sick_threshold = pred[i, 0] if (len(thresholds) == pred.shape[1] - 1) else np.minimum(pred[i, 0], thresholds[-1])

            if np.sum(pred[i,1:]) > is_sick_threshold:
                values = np.copy(pred[i,1:])

                for _ in range(len(values)):
                    max_ = np.argmax(values)
                    if values[max_] > thresholds[max_]:
                        pred_lb[i] = max_+1
                        break
                    # 'else' -- class with higher probability is still lower than its threshold
                    # so, others classes will be checked
                    values[max_] = -1

                # If no assignment was made although the sum of probabilities is higher than the probability of being healthy
                # Mark current image as the class that has the higher probability
                if pred_lb[i] == 0:
                    pred_lb[i] = np.argmax(pred[i,1:]) + 1

    else:
        pred_lb = np.where(pred[:,1] > thresholds[0], 1, 0)

    return pred_lb


def get_possible_dr_levels(predicted_class):
    return ','.join(list(map(str, list(DR_LEVELS_PER_CLASS[predicted_class])))) 


def prepare_image(path):

    # Read image
    image = cv2.imread(path)

    # Find retina
    size_x, size_y, radius, center_x, center_y = preprocess.get_data_image(path)

    if radius == -1:
        return None

    # Crop retina and resize it to 540x540
    image = preprocess.crop_and_resize_image(image, radius, center_x, center_y, size=image_size[0])

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def predict_image(image, model):

    # # Read image
    # image = cv2.imread(path)

    # # Find retina
    # size_x, size_y, radius, center_x, center_y = preprocess.get_data_image(path)

    # if radius == -1:
    #     return None, 'No se ha podido detectar el disco óptico'

    # # Crop retina and resize it to 540x540
    # image = preprocess.crop_and_resize_image(image, radius, center_x, center_y, size=image_size[0])

    # # Convert to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image = image.astype(np.float32) / 255.

    # Get model's prediction
    prediction = model(np.expand_dims(image, axis=0)).numpy()
    predicted_class = int(get_predicted_labels(prediction, acc_thresholds))

    dr_levels = get_possible_dr_levels(predicted_class)

    output = 'Sano' if predicted_class==0 else 'Indicios de Retinopatía Diabética'

    return prediction[0,...], output


def main():

    # Load model
    model = load_model(model_config, model_weights)

    while True:

        try:
            path = str(input('\n>>> Introduce ruta a la imagen (q para salir): '))

            if path.upper() == 'Q':
                break

            image = prepare_image(path)
            if image is not None:
                prediction, output = predict_image(image, model)

                print(prediction)
                print(output)

            else:
                print('No se detectó disco óptico')

        except Exception as e:
            print(e)

    pass


if __name__ == '__main__':
    main()