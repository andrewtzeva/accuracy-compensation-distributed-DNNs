from calibration import *
from accuracy_detection import *
from accuracy_refinement import accuracy_refinement
from utils.superclasses import super_map
from keras_applications import nasnet


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


def test():
    mobile_model = 'mobilenet'
    nasnet = keras.applications.nasnet.NASNetLarge(weights='imagenet')
    nasnet.summary()
    scaler_mobile = TemperatureScaling()

    path_mobile, val_file_mobile, test_file_mobile = load_model_files(mobile_model, scaler_mobile)

    #calibrate(scaler_mobile, path_mobile, val_file_mobile)
    predictions_mobile, labels = infer(scaler_mobile, path_mobile, test_file_mobile)

    acc_sum = 0
    for i, pred_distribution in enumerate(predictions_mobile):
        if bvsb(pred_distribution) < 0.165 or max_confidence(pred_distribution) < 0.54:
            pred_superclass = accuracy_refinement(pred_distribution)

            true_index = np.argmax(labels[i])
            true_superclass = super_map[true_index]

            if pred_superclass == true_superclass:
                acc_sum += 1
        else:
            true_class_index = np.argmax(labels[i])
            pred_class_index = np.argmax(pred_distribution)

            if true_class_index == pred_class_index:
                acc_sum += 1

    print(acc_sum/len(predictions_mobile))


test()
