import numpy as np
from .metrics import cross_entropy


def expected_cross_entropy_srv(server_predictions, client_predictions):
    cross_entropy_sum = 0
    num_of_samples = len(server_predictions)

    for i in range(num_of_samples):
        cross_entropy_sum += cross_entropy(server_predictions[i], client_predictions[i])

    return cross_entropy_sum/num_of_samples


def expected_confidence_diff(server_predictions, client_predictions, labels):
    confidence_diff_sum = 0
    num_of_samples = len(server_predictions)

    for i in range(num_of_samples):
        true_class_index = np.argmax(labels[i])
        server_confidence = server_predictions[i][true_class_index]
        client_confidence = client_predictions[i][true_class_index]

        confidence_diff_sum += abs(server_confidence - client_confidence)

    return confidence_diff_sum/num_of_samples
