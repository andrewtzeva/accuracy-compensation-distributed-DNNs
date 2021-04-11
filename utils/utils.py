import numpy as np


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


