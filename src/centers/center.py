# This class is used for generating files of training, test, and center data sets for each fold of our RBF network. All
# resulting files are stored under the "rbf-data" directory.

from src.centers.edited_knn import EditedKNN
from src.centers.k_means import KMeans
from src.centers.pam_nn import PamNN
import src.data.data_set as d
import math


def save_edited_knn_clusters(data, k, data_file_name):
    folds = data.validation_folds(10)
    for fold_i, fold in enumerate(folds):
        print(fold_i)
        test = fold['test']
        train = fold['train']
        eknn = EditedKNN(train, k)
        centers = eknn.training_data

        # We need to remove the centers from the original training data.
        updated_train_data = train.copy()
        for example in centers.data:
            if example in updated_train_data.data:
                updated_train_data.data.remove(example)

        # Save each test, train, and center data to their respective files.
        file_name = "../../rbf-data/" + data_file_name + "-eknn-fold-" + str(fold_i)
        test.save_file(file_name + "-test.txt")
        updated_train_data.save_file(file_name + "-train.txt")
        centers.save_file(file_name + "-centers.txt")


def save_k_means(data, k, data_file_name):
    folds = data.validation_folds(10)
    for fold_i, fold in enumerate(folds):
        print(fold_i)
        test = fold['test']
        train = fold['train']
        kmeans = KMeans(train, k)
        centers = test.copy()
        centers.data = kmeans.centroids

        # We don't need to worry about removing centers from the original training data since the clusters are not
        # points from the data.

        # Save each test, train, and center data to their respective files.
        file_name = "../../rbf-data/"  + data_file_name + "-kmeans-fold-" + str(fold_i)
        test.save_file(file_name + "-test.txt")
        train.save_file(file_name + "-train.txt")
        centers.save_file(file_name + "-centers.txt")


def save_pam(data, k, data_file_name):
    folds = data.validation_folds(10)
    for fold_i, fold in enumerate(folds):
        print(fold_i)
        test = fold['test']
        train = fold['train']
        pam = PamNN(train, k)
        centers = test.copy()
        centers.data = pam.medoids

        # We need to remove the centers from the original training data.
        updated_train_data = train.copy()
        for example in centers.data:
            if example in updated_train_data.data:
                updated_train_data.data.remove(example)

        # Save each test, train, and center data to their respective files.
        file_name = "../../rbf-data/" + data_file_name + "-pam-fold-" + str(fold_i)
        test.save_file(file_name + "-test.txt")
        train.save_file(file_name + "-train.txt")
        centers.save_file(file_name + "-centers.txt")


def run_eknn():
    abalone_data = d.get_abalone_data("../../data/abalone.data")
    abalone_k = math.ceil(math.sqrt(len(abalone_data.data)))
    save_edited_knn_clusters(abalone_data, abalone_k, "abalone")

    # car_data = d.get_car_data("../../data/car.data")
    # car_k = math.ceil(math.sqrt(len(car_data.data)))
    # save_edited_knn_clusters(car_data, car_k, "car")
    #
    # segmentation_data = d.get_segmentation_data("../../data/segmentation.data")
    # segmentation_k = math.ceil(math.sqrt(len(segmentation_data.data)))
    # save_edited_knn_clusters(segmentation_data, segmentation_k, "segmentation2")


def run_kmeans():
    abalone_data = d.get_abalone_data("../../data/abalone.data")
    abalone_k = 2510
    save_k_means(abalone_data, abalone_k, "abalone")

    car_data = d.get_car_data("../../data/car.data")
    car_k = 90
    save_k_means(car_data, car_k, "car")

    segmentation_data = d.get_segmentation_data("../../data/segmentation.data")
    segmentation_k = 43
    save_k_means(segmentation_data, segmentation_k, "segmentation")

    forest_fires_data = d.get_forest_fires_data("../../data/forestfires.data")
    forest_fires_k = math.ceil(len(forest_fires_data.data) / 4)
    save_k_means(forest_fires_data, forest_fires_k, "forestfires")

    machine_data = d.get_machine_data("../../data/machine.data")
    machine_k = math.ceil(len(machine_data.data) / 4)
    save_k_means(machine_data, machine_k, "machine")

    wine_data = d.get_wine_data("../../data/winequality.data")
    wine_k = math.ceil(len(wine_data.data) / 4)
    save_k_means(wine_data, wine_k, "winequality")


def run_pam():
    abalone_data = d.get_abalone_data("../../data/abalone.data")
    abalone_k = 2510
    save_pam(abalone_data, abalone_k, "abalone")

    car_data = d.get_car_data("../../data/car.data")
    car_k = 90
    save_pam(car_data, car_k, "car")

    segmentation_data = d.get_segmentation_data("../../data/segmentation.data")
    segmentation_k = 43
    save_pam(segmentation_data, segmentation_k, "segmentation")

    forest_fires_data = d.get_forest_fires_data("../../data/forestfires.data")
    forest_fires_k = math.ceil(len(forest_fires_data.data) / 4)
    save_pam(forest_fires_data, forest_fires_k, "forestfires")

    machine_data = d.get_machine_data("../../data/machine.data")
    machine_k = math.ceil(len(machine_data.data) / 4)
    save_pam(machine_data, machine_k, "machine")

    wine_data = d.get_wine_data("../../data/winequality.data")
    wine_k = math.ceil(len(wine_data.data) / 4)
    save_pam(wine_data, wine_k, "winequality")


# run_eknn()
# run_kmeans()
run_pam()
