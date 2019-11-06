# driver.py
# File for running our experimental design.
import src.data.data as d
import src.mfnn as mfnn


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


def main():
    # test_data = d.get_test_data().get_data()
    # training_set = test_data
    # validation_set = test_data * 2

    # multilayer = mfnn.MFNN(training_set, validation_set, [2, 2, 2], None, None, None)
    # example = test_data[0][:-1]
    # multilayer.backpropagation(example, [1,0])

    # We can run the RBF network using the following helper function:
    run_rbf_network("segmentation-eknn", d.get_segmentation_data)


