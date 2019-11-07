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
        self.training_data = training_data.copy()
        self.find_edited_data()

    # Updates the edit_training_data variable to the edited data_set, that is the data_set with unnecessary
    # vectors removed
    def find_edited_data(self):
        counter = 0
        length = len(self.training_data.data)
        while True:
            # print(str(counter) + "/" + str(length))
            if counter > length:
                break
            counter += 1

            # prev_accuracy = self.get_validation_accuracy()
            example = self.training_data.data.pop(0)
            # new_accuracy = self.get_validation_accuracy()
            if util.get_highest_class(self.run(example)) != example[self.training_data.class_col]:
                self.training_data.data.append(example)
            else:
                length -= 1
                counter = 0



