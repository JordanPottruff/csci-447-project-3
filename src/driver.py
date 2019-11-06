# driver.py
# File for running our experimental design.
import src.data.data_set as d
from src.networks.radial_basis_nn import RBFNN
from src.networks.mfnn import MFNN


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


def test_classification():
    test_data = d.get_classification_test_data()
    training_set = test_data.copy()
    training_set.data *= 12
    validation_set = training_set.copy()
    validation_set.data *= 10

    multilayer = MFNN(training_set, validation_set, [2, 2, 2], 1, None, [0, 1])
    multilayer.train()


def test_regression():
    test_data = d.get_regression_test_data()
    training_set = test_data.copy()
    training_set.data *= 12
    validation_set = training_set.copy()
    validation_set.data *= 10

    multilayer = MFNN(training_set, validation_set, [2, 10, 1], 1, None, None)
    multilayer.train()


def main():
    test_regression()
    # test_classification()
    # We can run the RBF network using the following helper function:
    # run_rbf_network("segmentation-eknn", d.get_segmentation_data)


main()