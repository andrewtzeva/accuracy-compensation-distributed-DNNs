import numpy as np
from keras.preprocessing import image
import keras


def reform_samples(predictions, labels):
    y_pred = []
    y_true = []

    for i in range(len(predictions)):
        pred_index = np.argmax(predictions[i])
        true_index = np.argmax(labels[i])

        y_pred.append(pred_index)
        y_true.append(true_index)

    return np.asarray(y_pred), np.asarray(y_true)


def one_hot_encode(test_labels, num_of_classes):
    labels = []

    for true_class_index in test_labels:
        label = np.zeros(num_of_classes)
        label[true_class_index] = 1
        labels.append(label)

    return np.asarray(labels)


def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


