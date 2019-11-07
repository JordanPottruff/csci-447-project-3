# driver.py
# File for running our experimental design.
import src.data.data_set as d
from src.networks.radial_basis_nn import RBFNN
from src.networks.mfnn import MFNN
import math


# Runs the RBF network by accessing the train, test, and center data from files. Files are used so that we do not have
# to recompute EditedKNN, K-Means, and PAM each time we want to run the RBF. Therefore, our cross validation is also
# already taken care of.
def run_rbf_network(data_set_name, data_opener):
    data_set_name = "../rbf-data/" + data_set_name
    for fold_i in range(10):
        train_data = data_opener(data_set_name + "-fold-" + str(fold_i) + "-train.txt", False)
        test_data = data_opener(data_set_name + "-fold-" + str(fold_i) + "-test.txt", False)
        center_data = data_opener(data_set_name + "-fold-" + str(fold_i) + "-centers.txt", False)

        # The training, test, and centers are now loaded in the respective variables above.
        outputs = []
        for row in center_data.data:
            outputs.append(row[center_data.class_col])
        outputs = list(set(outputs))
        # Set Learning Rate
        learning_rate = 1
        # Run RBFNN
        output_values = RBFNN(center_data, test_data, train_data, outputs, learning_rate).run_rbfnn()


def run_mfnn_classification(data_set, classes, layers, learning_rate, momentum):
    folds = data_set.validation_folds(10)
    for fold_i, fold in enumerate(folds):
        test = fold['test']
        train, validation = fold['train'].partition(.8)
        mfnn = MFNN(train, validation, layers, learning_rate, momentum, 100, classes)
        mfnn.train()
        print(mfnn.get_accuracy(test))


def run_mfnn_regression(data_set, layers, learning_rate, momentum):
    folds = data_set.validation_folds(10)
    for fold_i, fold in enumerate(folds):
        test = fold['test']
        train, validation = fold['train'].partition(.8)
        mfnn = MFNN(train, validation, layers, learning_rate, momentum, 100, None)
        mfnn.train()
        print(mfnn.get_error(test))


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


def main():
    abalone_data = d.get_abalone_data("../data/abalone.data")
    car_data = d.get_car_data("../data/car.data")
    segmentation_data = d.get_segmentation_data("../data/segmentation.data")

    forest_fires_data = d.get_forest_fires_data("../data/forestfires.data")
    machine_data = d.get_machine_data("../data/machine.data")
    wine_data = d.get_wine_data("../data/winequality.data")

    # test_regression()
    # test_classification()
    # We can run the RBF network using the following helper function:
    # run_rbf_network("segmentation-eknn", d.get_segmentation_data)

    # MFNN Classifications
    # run_mfnn_classification(car_data, ["unacc", "acc", "good", "vgood"], [6, 10, 4], 1, 0.01)

    # MFNN Regressions
    # run_mfnn_regression(wine_data, [11, 5, 1], .05, 0.1)
    run_mfnn_regression(machine_data, [6, 2, 1], .001, .2)
    # run_mfnn_regression(forest_fires_data, [12, 10, 1], .01, 0.1)



main()