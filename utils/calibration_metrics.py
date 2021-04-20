import numpy as np


def get_predictions(predictions, labels):
    num_of_samples = len(predictions)
    preds = np.zeros(num_of_samples)
    confs = np.zeros(num_of_samples)

    for i in range(num_of_samples):
        true_class_index = np.argmax(labels[i])
        pred_class_index = np.argmax(predictions[i])

        if true_class_index == pred_class_index:
            preds[i] = 1

        confs[i] = predictions[i][pred_class_index]

    return preds, confs


def get_predictions_with_class(predictions, labels):
    num_of_samples = len(predictions)
    num_of_classes = len(labels[0])
    preds = np.zeros(num_of_samples)
    confs = np.zeros(num_of_samples)
    classes = np.zeros(num_of_samples)

    for i in range(num_of_samples):
        true_class_index = np.argmax(labels[i])
        pred_class_index = np.argmax(predictions[i])

        if true_class_index == pred_class_index:
            preds[i] = 1

        confs[i] = predictions[i][pred_class_index]
        classes[i] = true_class_index

    return preds, confs, classes, num_of_classes


def binwise_confindence_accuracy_diff(preds, confs, low, high):
    num_of_samples = len(confs)
    indexes = []

    for i in range(num_of_samples):
        if low <= confs[i] <= high:
            indexes.append(i)

    acc_sum, conf_sum = 0, 0

    for i in range(len(indexes)):
        acc_sum += preds[indexes[i]]
        conf_sum += confs[indexes[i]]

    acc = acc_sum/len(indexes) if len(indexes) != 0 else 0
    conf = conf_sum/len(indexes) if len(indexes) != 0 else 0

    return abs(acc - conf)


def weighted_binwise_confindence_accuracy_diff(preds, confs, low, high):
    num_of_samples = len(confs)
    indexes = []

    for i in range(num_of_samples):
        if low <= confs[i] <= high:
            indexes.append(i)

    acc_sum, conf_sum = 0, 0

    for i in range(len(indexes)):
        acc_sum += preds[indexes[i]]
        conf_sum += confs[indexes[i]]

    acc = acc_sum/len(indexes) if len(indexes) != 0 else 0
    conf = conf_sum/len(indexes) if len(indexes) != 0 else 0

    return len(indexes) * abs(acc - conf)


def weighted_binwise_confindence_accuracy_diff_per_class(preds, confs, classes, low, high, k):
    num_of_samples = len(confs)
    indexes = []

    for i in range(num_of_samples):
        if low <= confs[i] <= high and classes[i] == k:
            indexes.append(i)

    acc_sum, conf_sum = 0, 0

    for i in range(len(indexes)):
        acc_sum += preds[indexes[i]]
        conf_sum += confs[indexes[i]]

    acc = acc_sum/len(indexes) if len(indexes) != 0 else 0
    conf = conf_sum/len(indexes) if len(indexes) != 0 else 0

    return len(indexes) * abs(acc - conf)


def ece(preds, confs, num_of_bins = 10):
    step = 1/num_of_bins
    num_of_samples = len(preds)
    ece_sum = 0

    for i in range(num_of_bins):
        low = i*step
        high = (i+1)*step

        weighted_binwise_acc_conf_diff = weighted_binwise_confindence_accuracy_diff(preds, confs, low, high)
        ece_sum += weighted_binwise_acc_conf_diff

    return ece_sum/num_of_samples


def mce(preds, confs, num_of_bins=15):
    step = 1/num_of_bins
    diff_values = []

    for i in range(num_of_bins):
        low = i*step
        high = (i+1)*step

        binwise_acc_conf_diff = binwise_confindence_accuracy_diff(preds, confs, low, high)
        diff_values.append(binwise_acc_conf_diff)

    return max(diff_values)


def sce(preds, confs, classes, num_of_classes, num_of_bins=15):
    step = 1 / num_of_bins
    num_of_samples = len(preds)
    class_sum = 0

    for k in range(num_of_classes):
        ece_sum = 0
        for i in range(num_of_bins):
            low = i * step
            high = (i + 1) * step

            weighted_binwise_acc_conf_diff = weighted_binwise_confindence_accuracy_diff_per_class(preds, confs, classes, low, high, k)
            ece_sum += weighted_binwise_acc_conf_diff

        class_sum += ece_sum

    return class_sum/(num_of_samples * num_of_classes)
