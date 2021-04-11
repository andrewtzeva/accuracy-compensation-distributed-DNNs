import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from .utils import reform_samples


def top1_accuracy(predictions, labels):
    acc_sum = 0
    num_of_samples = len(predictions)

    for i in range(num_of_samples):
        true_class_index = np.argmax(labels[i])
        pred_class_index = np.argmax(predictions[i])

        if true_class_index == pred_class_index:
            acc_sum += 1

    return acc_sum/num_of_samples


def topk_accuracy(predictions, labels, k):
    acc_sum = 0
    num_of_samples = len(predictions)

    for i in range(num_of_samples):
        pred_distribution = predictions[i]
        true_class_index = np.argmax(labels[i])
        sorted_args = pred_distribution.argsort()[-k:][::-1]

        if true_class_index in sorted_args:
            acc_sum += 1

    return acc_sum/num_of_samples


def expected_accuracy(predictions, labels):
    acc_sum = 0
    num_of_samples = len(predictions)

    for i in range(num_of_samples):
        true_class_index = np.argmax(labels[i])
        prob_estimate = predictions[i][true_class_index]

        acc_sum += prob_estimate

    return acc_sum/num_of_samples


def entropy(pred_distribution):
    H = 0

    for prob_estimate in pred_distribution:
        H += -prob_estimate * np.log(prob_estimate)

    return H


def expected_entropy(predictions):
    entropy_sum = 0
    num_of_samples = len(predictions)

    for prediction in predictions:
        entropy_sum += entropy(prediction)

    return entropy_sum/num_of_samples


def bvsb(pred_distribution):
    sorted_args = pred_distribution.argsort()[-2:][::-1]
    best_prob_estimate = pred_distribution[sorted_args[0]]
    second_best_prob_estimate = pred_distribution[sorted_args[1]]

    return best_prob_estimate - second_best_prob_estimate


def expected_bvsb(predictions):
    bvsb_sum = 0
    num_of_samples = len(predictions)

    for prediction in predictions:
        bvsb_sum += bvsb(prediction)

    return bvsb_sum/num_of_samples


def gbvsb(pred_distribution, M, N):
    sorted_args_1 = pred_distribution.argsort()[-M:][::-1]
    pred_distribution_rest = [v for i, v in enumerate(pred_distribution) if i not in sorted_args_1]

    sorted_args_2 = pred_distribution_rest.argsort()[-N:][::-1]

    prob_sum_1, prob_sum_2 = 0, 0

    for i in sorted_args_1:
        prob_sum_1 += pred_distribution[i]
    for i in sorted_args_2:
        prob_sum_2 += pred_distribution[i]

    return prob_sum_1 - prob_sum_2


def gini_index(pred_distribution):
    gini = 1

    for prob_estimate in pred_distribution:
        gini -= prob_estimate * prob_estimate

    return gini


def expected_gini_index(predictions):
    gini_sum = 0
    num_of_samples = len(predictions)

    for prediction in predictions:
        gini_sum += gini_index(prediction)

    return gini_sum/num_of_samples


def cross_entropy(pred_distribution, true_distribution):
    H = 0

    for i in range(len(pred_distribution)):
        H += -true_distribution[i] * np.log(pred_distribution[i])

    return H


def expected_cross_entropy(predictions, labels):
    cross_entropy_sum = 0
    num_of_samples = len(predictions)

    for i in range(num_of_samples):
        cross_entropy_sum += cross_entropy(predictions[i], labels[i])

    return cross_entropy_sum/num_of_samples


def balanced_accuracy(predictions, labels):
    y_pred, y_true = reform_samples(predictions, labels)

    return balanced_accuracy_score(y_true, y_pred)


def f1_score(predictions, labels):
    y_pred, y_true = reform_samples(predictions, labels)

    return f1_score(y_true, y_pred, average='macro')






