from .utils import *
from utils.metrics import *
from scipy.optimize import minimize
from os.path import join

"""
Calculated thresholds for sending 1/4 #samples to server: BVSB=0.16582, MAX_CONFIDENCE=0.54031

"""


def objective_calc(thres, predictions_server, predictions_mobile, labels):
    count_server = 0
    predictions = []

    for i, pred in enumerate(predictions_mobile):
        if bvsb(pred) < thres[0] or max_confidence(pred) < thres[1]:
            predictions.append(predictions_server[i])
            count_server += 1
        else:
            predictions.append(predictions_mobile[i])

    predictions = np.asarray(predictions)
    acc = top1_accuracy(predictions, labels)
    error = 1 - acc

    return error, count_server


def optimize_thresholds(max_samples_to_server=5000, verbose=False):
    server_model = 'nasnet_large'
    mobile_model = 'mobilenet'

    scaler_server = ""
    scaler_mobile = ""

    path_server, val_file_server, test_file_server = load_model_files(server_model, scaler_server)
    path_mobile, val_file_mobile, test_file_mobile = load_model_files(mobile_model, scaler_mobile)

    test_file_mobile = join(path_mobile, test_file_mobile)
    test_file_server = join(path_server, test_file_server)

    predictions_mobile, labels = unpickle_predictions(test_file_mobile)
    predictions_server, labels = unpickle_predictions(test_file_server)
    labels = one_hot_encode(labels, 1000)

    def objective(thres, predictions_server, predictions_mobile, labels):
        error = objective_calc(thres, predictions_server, predictions_mobile, labels)[0]
        print(1-error)
        return error

    def constraint(thres, predictions_server, predictions_mobile, labels):
        count_server = objective_calc(thres, predictions_server, predictions_mobile, labels)[1]
        print(count_server)
        return max_samples_to_server-count_server

    con = {'type': 'ineq', 'fun': constraint, 'args': (predictions_server, predictions_mobile, labels)}

    sol = minimize(objective, x0=[0.1, 0.3], bounds=((0.0, 1.0), (0.0, 1.0)),
                   args=(predictions_server, predictions_mobile, labels), method='COBYLA', options={'rhobeg': 0.04},
                   constraints=[con])

    if verbose:
        print(sol)

    return sol
