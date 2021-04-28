from calibration import *
from accuracy_detection import *


def main():
    server_model = 'nasnet_large'
    mobile_model = 'mobilenet'

    scaler_server = TemperatureScaling()
    scaler_mobile = TemperatureScaling()

    path_server, val_file_server, test_file_server = load_model_files(server_model, scaler_server)
    path_mobile, val_file_mobile, test_file_mobile = load_model_files(mobile_model, scaler_mobile)

    calibrate(scaler_server, path_server, val_file_server)
    predictions_server, labels = infer(scaler_server, path_server, test_file_server)

    calibrate(scaler_mobile, path_mobile, val_file_mobile)
    predictions_mobile, labels = infer(scaler_mobile, path_mobile, test_file_mobile)

    evaluate_accuracy_sys(mobile_model, server_model, predictions_server, predictions_mobile, labels, True)
    evaluate_accuracy_algo(mobile_model, predictions_mobile, labels, True)


main()
