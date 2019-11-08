# driver.py
# File for running our experimental design.
import src.data.data_set as d
from src.networks.radial_basis_nn import RBFNN
from src.networks.mfnn import MFNN
import math


def test_classification():
    test_data = d.get_classification_test_data()
    training_set = test_data.copy()
    training_set.data *= 12
    validation_set = training_set.copy()
    validation_set.data *= 10
    multilayer = MFNN(training_set, validation_set, [2, 2, 2], 1, 0.01, 100, [0, 1])
    multilayer.train()


def test_regression():
    test_data = d.get_regression_test_data()
    training_set = test_data.copy()
    training_set.data *= 12
    validation_set = training_set.copy()
    validation_set.data *= 10

    multilayer = MFNN(training_set, validation_set, [2, 10, 1], 1, 0.01, 100, None)
    multilayer.train()


def get_rbf_data(data_set_name, data_opener):
    data_set_name = "../rbf-data/" + data_set_name
    folds = []
    for fold_i in range(10):
        train_data = data_opener(data_set_name + "-fold-" + str(fold_i) + "-train.txt", False)
        test_data = data_opener(data_set_name + "-fold-" + str(fold_i) + "-test.txt", False)
        center_data = data_opener(data_set_name + "-fold-" + str(fold_i) + "-centers.txt", False)
        folds.append({'train': train_data, 'test': test_data, 'center': center_data})
    return folds


def run_rbfnn_classification(data_set_name, data_opener, classes, learning_rate, convergence_size):
    print(data_set_name)
    # Standard parameters to run on the data_set
    num_folds = 10
    folds = get_rbf_data(data_set_name, data_opener)
    fold_average = []
    for fold_i, fold in enumerate(folds):
        test = fold['test']
        train, validation = fold['train'].partition(.8)
        center = fold['center']
        size_inputs = len(test.attr_cols)
        rbfnn = RBFNN(center, train, validation, size_inputs, learning_rate, convergence_size, classes)
        rbfnn.train()
        fold_accuracy = rbfnn.get_accuracy(test)
        fold_average.append(fold_accuracy)
        print("\t Fold: " + str(fold_i + 1) + " - Accuracy: " + str(fold_accuracy))
    print("Average Accuracy: " + str(sum(fold_average) / num_folds))


def run_rbfnn_regression(data_set_name, data_opener, learning_rate, convergence_size):
    print(data_set_name)
    # Standard parameters to run on the data_set
    num_folds = 10
    folds = get_rbf_data(data_set_name, data_opener)
    fold_average = []
    for fold_i, fold in enumerate(folds):
        test = fold['test']
        train, validation = fold['train'].partition(.8)
        center = fold['center']
        size_inputs = len(test.attr_cols)
        rbfnn = RBFNN(center, train, validation, size_inputs, learning_rate, convergence_size)
        rbfnn.train()
        fold_error = rbfnn.get_error(test)
        fold_average.append(fold_error)
        print("\t Fold: " + str(fold_i + 1) + " - Error: " + str(fold_error))
    print("Average Error: " + str(sum(fold_average) / num_folds))


def run_mfnn_classification(data_set, classes, learning_rate, momentum, convergence_size):
    print(data_set.filename)
    # Standard parameters to run on the data_set
    num_folds = 10
    size_inputs = len(data_set.attr_cols)
    num_hidden_layers = [0, 1, 2]
    size_outputs = len(classes)
    size_hidden_layers = math.floor((size_inputs + size_outputs) / 2)
    folds = data_set.validation_folds(num_folds)
    for hidden_layer in num_hidden_layers:
        layers = [size_inputs] + ([size_hidden_layers] * hidden_layer) + [size_outputs]
        print("Network " + str(layers))
        fold_average = []
        for fold_i, fold in enumerate(folds):
            test = fold['test']
            train, validation = fold['train'].partition(.8)
            print(layers)
            mfnn = MFNN(train, validation, layers, learning_rate, momentum, convergence_size, classes)
            mfnn.train()
            fold_average.append(mfnn.get_accuracy(test))
            print("\t Fold: " + str(fold_i+1) + " - Accuracy: " + str(mfnn.get_accuracy(test)))
        print("Average Accuracy: " + str(sum(fold_average) / num_folds))


def run_mfnn_regression(data_set, learning_rate, momentum, convergence_size):
    print(data_set.filename)
    num_folds = 10
    size_inputs = len(data_set.attr_cols)
    num_hidden_layers = [0, 1, 2]
    size_hidden_layers = math.floor((size_inputs + 1) / 2)

    folds = data_set.validation_folds(num_folds)
    for hidden_layers in num_hidden_layers:
        layers = [size_hidden_layers] * (hidden_layers + 2)
        layers[0], layers[-1] = size_inputs, 1
        print("Network " + str(layers))
        fold_average = []
        for fold_i, fold in enumerate(folds):
            test = fold['test']
            train, validation = fold['train'].partition(.8)
            mfnn = MFNN(train, validation, layers, learning_rate, momentum, convergence_size, None)
            mfnn.train()
            fold_average.append(mfnn.get_error(test))
            print("\t Fold: " + str(fold_i+1) + " - Error: " + str(mfnn.get_error(test)))
        print("Average Error: " + str(sum(fold_average)/num_folds))


def run_rbfnn_classification_data_sets(center_alg_name):
    abalone_classes = [float(i) for i in range(1, 30)]
    car_classes = ["unacc", "acc", "good", "vgood"]
    segmentation_classes = ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"]

    run_rbfnn_classification("abalone-" + center_alg_name, d.get_abalone_data, abalone_classes, 1, 10)
    run_rbfnn_classification("car-" + center_alg_name, d.get_car_data, car_classes, 1, 10)
    run_rbfnn_classification("segmentation-" + center_alg_name, d.get_segmentation_data, segmentation_classes, 1, 30)


def run_rbfnn_regression_data_sets(center_alg_name):
    run_rbfnn_regression("forestfires-" + center_alg_name, d.get_forest_fires_data, 1, 100)
    run_rbfnn_regression("machine-" + center_alg_name, d.get_machine_data, .1, 100)
    # run_rbfnn_regression("winequality-" + center_alg_name, d.get_wine_data, 1, 20)


def run_mfnn_classification_data_sets():
    abalone_data = d.get_abalone_data("../data/abalone.data")
    abalone_classes = [float(i) for i in range(1, 30)]
    car_data = d.get_car_data("../data/car.data")
    car_classes = ["unacc", "acc", "good", "vgood"]
    segmentation_data = d.get_segmentation_data("../data/segmentation.data")
    segmentation_classes = ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"]

    # run_mfnn_classification(abalone_data, abalone_classes, 1, .1, 100)
    run_mfnn_classification(car_data, car_classes, 1, .1, 100)
    run_mfnn_classification(segmentation_data, segmentation_classes, 1, .1, 100)


def run_mfnn_regression_data_sets():
    forest_fires_data = d.get_forest_fires_data("../data/forestfires.data")
    machine_data = d.get_machine_data("../data/machine.data")
    wine_data = d.get_wine_data("../data/winequality.data")

    run_mfnn_regression(forest_fires_data, 1, 0.1, 100)
    run_mfnn_regression(machine_data, 1, 0.1, 100)
    run_mfnn_regression(wine_data, 1, 0.1, 100)


def main():
    learning_rate = 0.00001
    momentum = 0.2
    convergence_size = 100

    # run_mfnn_regression_data_sets()
    # run_mfnn_classification_data_sets()

    # run_rbfnn_regression_data_sets("pam")
    run_rbfnn_classification_data_sets("kmeans")

main()
