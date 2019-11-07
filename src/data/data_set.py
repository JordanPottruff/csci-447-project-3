from typing import Dict
import math
import random
import numpy as np
import src.util as util

ABALONE_DATA_FILE = "../data/abalone.data"
CAR_DATA_FILE = "../data/car.data"
FOREST_FIRE_DATA_FILE = "../data/forestfires.data"
MACHINE_DATA_FILE = "../data/machine.data"
SEGMENTATION_DATA_FILE = "../data/segmentation.data"
WINE_DATA_FILE = "../data/winequality.data"
CLASSIFICATION_TEST_DATA_FILE = '../data/test/classification_test_set.data'
REGRESSION_TEST_DATA_FILE = '../data/test/regression_test_set.data'


class DataSet:

    # Creates a DataSet object from:
    # - data: a list of examples, where each example is either a list or tuple.
    # - attr_cols: a list of ints where each int is the index of a column representing an attribute.
    # - class_col: an int that is the index of the column representing the class value.
    # - num_classes: the number of classes in the data (including classes that may not be present in the data list).
    # - filename: (optional) string denoting the name of the file the data is originally sourced from.
    def __init__(self, data: list, attr_cols: list, class_col: int, filename=""):
        self.data = data
        self.attr_cols = attr_cols
        self.class_col = class_col
        self.filename = filename

    def copy(self):
        return DataSet(self.data.copy(), self.attr_cols.copy(), self.class_col, self.filename)

    def get_data(self):
        return self.data.copy()

    def get_len(self):
        return len(self.data)

    # Removes the first 'length' rows from the data. Use if there is header information.
    def remove_header(self, length: int):
        self.data = self.data[length:]

    # Used to handle data sets that involve discrete attribute values. The values in the attribute at the specified
    # column are converted using the given map from the original value to the new value. This is purposefully abstract
    # in order for the Data class to work with numerous data sets.
    def convert_str_attribute(self, col: int, value_map: Dict[str, float]):
        for i in range(len(self.data)):
            row = list(self.data[i])
            if row[col] in value_map:
                row[col] = value_map[row[col]]
            self.data[i] = tuple(row)

    # Converts values in a specified set of columns (represented as indices) to floating point values.
    def convert_to_float(self, cols):
        for line_i in range(len(self.data)):
            line = self.data[line_i]
            new_line = list(line)
            for i in range(len(line)):
                if i not in cols:
                    continue
                new_line[i] = float(line[i])
            line = tuple(new_line)
            self.data[line_i] = line

    # Normalizes the values in a specified set of columns (represented as indices) to a z-score. For interpretation, the
    # new value in the data set represents how many standard deviations an attribute value is from the mean of the
    # attribute.
    def normalize_z_score(self, cols):
        for col in cols:
            # Calculate the mean of the column's data.
            col_sum = 0
            for row in self.data:
                col_sum += row[col]
            mean = col_sum / len(self.data)

            # Calculate the standard deviation of the column's data.
            sum_square_diffs = 0
            for row in self.data:
                sum_square_diffs += (mean - row[col]) ** 2
            standard_deviation = math.sqrt(sum_square_diffs)

            # Replace each column value with it's z-score.
            for row_i in range(len(self.data)):
                row = list(self.data[row_i])
                # If the standard deviation is 0, we just assign the z_score as 0 (no variation from mean).
                z_score = 0
                if standard_deviation != 0:
                    # Otherwise we can use the standard calculation for z_scores.
                    z_score = (row[col] - mean) / standard_deviation
                row[col] = z_score
                self.data[row_i] = tuple(row)

    # Shuffles the rows in the data randomly.
    def shuffle(self):
        random.shuffle(self.data)

    # Partitions the data set into two 2D lists. The first_percentage parameter specifies what proportion of
    # observations should fall into the first 2D list.
    def partition(self, first_percentage):
        cutoff = math.floor(first_percentage * len(self.data))
        first = DataSet(self.data[:cutoff], self.class_col, self.attr_cols)
        second = DataSet(self.data[cutoff:], self.class_col, self.attr_cols)
        return first, second

    # Creates n-"folds" of our data set, which can be used for cross validation. Each fold has a test set, containing
    # 1/n of the data, and a training set, containing (n-1)/n of the data. Returns the list of folds.
    def validation_folds(self, n):
        avg_size = len(self.data) / n
        sections = []
        for i in range(n):
            section_data = None
            # If we are in the final section, we make sure to take all elements to the very end of the data.
            if i == n - 1:
                sections.append(self.data[math.floor(avg_size * i):])
            # Otherwise, we use a normal range to create the data for the section.
            else:
                sections.append(self.data[math.floor(avg_size * i):math.floor(avg_size * (i + 1))])

        folds = [{} for i in range(n)]
        for i in range(n):
            folds[i]['test'] = DataSet(sections[i], self.attr_cols, self.class_col)
            train = []
            for j in range(n):
                if i != j:
                    train += sections[j]
            folds[i]['train'] = DataSet(train, self.attr_cols, self.class_col)
        return folds

    # Used to randomly sample our data to only be of length k.
    def sample(self, k):
        self.data = random.sample(self.data, k)

    # Save data to a file with the specified name.
    def save_file(self, file_name):
        file = open(file_name, "w")
        for line in self.data:
            file.write(",".join(map(str, line)) + "\n")
        file.close()

    # Prints the data set nicely.
    def print(self):
        for row in self.data:
            print(row)
        print()

    def distance(self, example_a: list, example_b: list):
        squared_sum = 0
        for attr_col in self.attr_cols:
            squared_sum += (example_a[attr_col] - example_b[attr_col]) ** 2
        return math.sqrt(squared_sum)

    def get_max_distance(self):
        max_distance = 0
        idx = 0
        starter_row = self.data[idx]
        while idx < len(self.data)-2:
            for i in range(idx+1, len(self.data)-1):
                # Calculate distance between starter_row
                distance = self.distance(starter_row, self.data[i])
                if distance > max_distance:
                    max_distance = distance
            idx += 1
            starter_row = self.data[idx]
        return max_distance

    # Returns a list of tuples, where each tuple stores (@0) the numpy representation of each example's attributes, and
    # (@1) the example's class.
    def get_numpy_list(self):
        numpy_list = []
        for example in self.data:
            attr_only = []
            for col in self.attr_cols:
                attr_only.append(example[col])
            numpy_list.append((np.array(attr_only), example[self.class_col]))
        return numpy_list




# NOTE:
# The following functions are meant to handle the preprocessing of the data sets used in our experimental design.


def get_abalone_data():
    return get_abalone_data(ABALONE_DATA_FILE)


# Gets the abalone data set.
def get_abalone_data(file_name, normalize=True):
    data = util.read_file(file_name)
    abalone_data = DataSet(data, list(range(1, 8)), 8, file_name)
    # Convert attribute columns to floats
    abalone_data.convert_to_float(list(range(1, 9)))
    # Normalize values
    if normalize:
        abalone_data.normalize_z_score(list(range(1, 8)))
    # Randomly shuffle values.
    abalone_data.shuffle()
    return abalone_data


def get_car_data():
    return get_car_data(CAR_DATA_FILE)


# Gets the car data set.
def get_car_data(file_name, normalize=True):
    data = util.read_file(file_name)
    car_data = DataSet(data, list(range(0, 6)), 6, file_name)
    # Convert attribute columns to numeric scheme
    car_data.convert_str_attribute(0, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    car_data.convert_str_attribute(1, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    car_data.convert_str_attribute(2, {'2': 2, '3': 3, '4': 4, '5more': 5})
    car_data.convert_str_attribute(3, {'2': 2, '4': 4, 'more': 5})
    car_data.convert_str_attribute(4, {'small': 0, 'med': 1, 'big': 2})
    car_data.convert_str_attribute(5, {'low': 0, 'med': 1, 'high': 2})
    # Normalize values.
    if normalize:
        car_data.normalize_z_score(list(range(0, 6)))
    # Randomly shuffle values.
    car_data.shuffle()
    return car_data


def get_forest_fires_data():
    return get_forest_fires_data(FOREST_FIRE_DATA_FILE)


# Gets the forest fires data set.
def get_forest_fires_data(file_name, normalize=True):
    data = util.read_file(file_name)
    forest_fires_data = DataSet(data, [0, 1] + list(range(4, 12)), 12, file_name)
    # Remove the first line, which is the header info.
    forest_fires_data.remove_header(1)
    # Convert applicable columns to floats, including the class column.
    forest_fires_data.convert_to_float([0, 1] + list(range(4, 13)))
    # Normalize values.
    if normalize:
        forest_fires_data.normalize_z_score([0, 1] + list(range(4, 12)))
    # Randomly shuffle values.
    forest_fires_data.shuffle()
    return forest_fires_data


def get_machine_data():
    return get_machine_data(MACHINE_DATA_FILE)


# Gets the machine data set.
def get_machine_data(file_name, normalize=True):
    data = util.read_file(file_name)
    # There is another final column but we probably want to exclude it.
    machine_data = DataSet(data, list(range(2, 8)), 8, file_name)
    # Convert all columns except the first two to floats, including the class column.
    machine_data.convert_to_float(list(range(2, 9)))
    # Normalize values.
    if normalize:
        machine_data.normalize_z_score(list(range(2, 8)))
    # Randomly shuffle values.
    machine_data.shuffle()
    return machine_data


def get_segmentation_data():
    return get_segmentation_data(SEGMENTATION_DATA_FILE)


# Gets the segmentation data set.
def get_segmentation_data(file_name, normalize=True):
    data = util.read_file(file_name)
    # Attribute columns are all numeric
    #  * Attribute #7, vedge-sd, removed because it is a standard deviation.
    #  * Attribute #9, hedge-sd, removed for same reason.
    attr_cols = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    segmentation_data = DataSet(data, attr_cols, 0, file_name)
    # Remove the first 5 lines, which is reserved for the header.
    segmentation_data.remove_header(5)
    # Convert all attribute columns to numeric values.
    segmentation_data.convert_to_float(attr_cols)
    # Normalize values.
    if normalize:
        segmentation_data.normalize_z_score(attr_cols)
    # Randomly shuffle values.
    segmentation_data.shuffle()
    return segmentation_data


def get_wine_data():
    return get_wine_data(WINE_DATA_FILE)


# Gets the wine data set.
def get_wine_data(file_name, normalize=True):
    data = util.read_file(file_name)
    wine_data = DataSet(data, list(range(0, 11)), 11, file_name)
    # Convert all attribute columns to numeric values.
    wine_data.convert_to_float(list(range(0, 12)))
    # Normalize values.
    if normalize:
        wine_data.normalize_z_score(list(range(0, 11)))
    # Randomly shuffle values.
    wine_data.shuffle()
    return wine_data


# Gets test_set data
def get_classification_test_data():
    data = util.read_file(CLASSIFICATION_TEST_DATA_FILE)
    test_data = DataSet(data, [0, 1], 2, CLASSIFICATION_TEST_DATA_FILE)
    test_data.convert_to_float([0, 1, 2])
    return test_data


def get_regression_test_data():
    data = util.read_file(REGRESSION_TEST_DATA_FILE)
    test_data = DataSet(data, [0, 1], 2, REGRESSION_TEST_DATA_FILE)
    test_data.remove_header(1)
    test_data.convert_to_float([0, 1, 2])
    test_data.normalize_z_score([0, 1])
    return test_data
