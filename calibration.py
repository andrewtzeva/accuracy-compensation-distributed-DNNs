import numpy as np
from utils.calibration_metrics import *
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from utils.utils import *
from os.path import join


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


class TemperatureScaling:

    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def gettemp(self):
        return self.temp

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        opt = minimize(self._loss_fun, x0=1, args=(logits, true), options={'maxiter': self.maxiter}, method=self.solver)
        self.temp = opt.x[0]

        return opt

    def predict(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)


def evaluate(predictions, labels, verbose=False, bins=15):
    """
    Evaluate model using various scoring measures: ECE, MCE, NLL, SCE

    Params:
        predictions: a list containing confidences for all the classes with a shape of (samples, classes)
        labels: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (ece, mce, loss), returns various scoring measures
    """

    preds, confs = get_predictions(predictions, labels)

    # Calculate ECE
    ece_val = ece(preds, confs, bins)
    # Calculate MCE
    mce_val = mce(preds, confs, bins)
    # Calculate NLL
    loss = log_loss(labels, predictions)

    if verbose:
        print("ECE:", ece_val)
        print("MCE:", mce_val)
        print("NLL:", loss)

    return ece_val, mce_val, loss


def calibrate(scaler, path, file):
    """
    Calibrate models scores, using output from logits files.

    Params:
        path (string): path to the folder with logits files
        files (string): pickled logits files (logits_val, y_val)
        m_kwargs (dictionary): keyword arguments for the calibration class initialization

    """

    f_path = join(path, file)
    (logits_val, y_val) = unpickle_predictions(f_path)

    y_val = np.asarray(y_val)
    y_val = one_hot_encode(y_val, 1000)

    scaler.fit(logits_val, y_val)


def infer(scaler, path, file, verbose=False):
    """
    Infer models predictions, using output from logits files.

    Params:
        path (string): path to the folder with logits files
        file (string): pickled logits files (logits_val, y_val)
        verbose (Bool): calibration scores printed out (default = False)

    Returns:
        predictions: calibrated probabilities (nd.array with shape [samples, classes])
        y_test: a list containing the actual class labels

    """

    f_path = join(path, file)
    (logits_test, y_test) = unpickle_predictions(f_path)

    y_test = one_hot_encode(y_test, 1000)
    predictions = scaler.predict(logits_test)

    if verbose:
        print("ECE %f; MCE %f; NLL %f" % evaluate(predictions, y_test, verbose=False))

    return predictions, y_test
