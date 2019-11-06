# edited_knn.py
# Implementation of edited KNN algorithm, which we create as a subclass of the KNN class.

from src.centers.knn import KNN
import src.util as util


# Edited KNN algorithm. Training data is automatically reduced. To classify, use KNN's "run" method.
class EditedKNN(KNN):

    # Upon creation of the model, the training data is reduced using the find_edited_data method. This allows us to
    # use the "run" method from KNN once the object is created.
    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        data_copy = training_data.copy().partition(.90)
        self.training_data = data_copy[0]
        self.validation_data = data_copy[1]
        self.find_edited_data()

    def get_validation_accuracy(self):
        correct = 0
        for example in self.validation_data.data:
            result = self.run(example)
            if not result:
                continue
            actual_class = util.get_highest_class(result)
            expected_class = example[self.validation_data.class_col]
            if actual_class == expected_class:
                correct += 1
        return correct / len(self.validation_data.data)

    # Updates the edit_training_data variable to the edited data_set, that is the data_set with unnecessary
    # vectors removed
    def find_edited_data(self):
        counter = 0

        last_cycles_accuracy = self.get_validation_accuracy()
        while last_cycles_accuracy <= self.get_validation_accuracy() + .01:
            training_data_copy = self.training_data.copy()
            for i in range(len(training_data_copy.data)):
                print(str(i) + "/" + str(len(training_data_copy.data)))
                prev_accuracy = self.get_validation_accuracy()
                example = training_data_copy.data[i]

                self.training_data.data.remove(example)

                if util.get_highest_class(self.run(example)) != example[self.training_data.class_col]:
                    self.training_data.data.append(example)
                else:
                    if prev_accuracy > self.get_validation_accuracy() + .01:
                        self.training_data.data.append(example)
            last_cycles_accuracy = self.get_validation_accuracy()










