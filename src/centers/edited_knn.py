# edited_knn.py
# Implementation of edited KNN algorithm, which we create as a subclass of the KNN class.

from src.centers.knn import KNN
import src.util as util
import math


# Edited KNN algorithm. Training data is automatically reduced. To classify, use KNN's "run" method.
class EditedKNN(KNN):

    # Upon creation of the model, the training data is reduced using the find_edited_data method. This allows us to
    # use the "run" method from KNN once the object is created.
    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        data_copy = training_data.copy().partition(.95)
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
    # def find_edited_data(self):
    #     counter = 0
    #
    #     last_cycles_accuracy = self.get_validation_accuracy()
    #     while last_cycles_accuracy <= self.get_validation_accuracy() + .01:
    #         counter += 1
    #         training_data_copy = self.training_data.copy()
    #         for i in range(len(training_data_copy.data)):
    #             print(str(i) + "/" + str(len(training_data_copy.data)))
    #             prev_accuracy = self.get_validation_accuracy()
    #             example = training_data_copy.data[i]
    #
    #             self.training_data.data.remove(example)
    #
    #             if util.get_highest_class(self.run(example)) != example[self.training_data.class_col]:
    #                 self.training_data.data.append(example)
    #             else:
    #                 if prev_accuracy > self.get_validation_accuracy() + .01:
    #                     self.training_data.data.append(example)
    #         last_cycles_accuracy = self.get_validation_accuracy()

    def find_edited_data(self):
        prev_size = float("inf")
        while len(self.training_data.data) < prev_size:
            prev_size = len(self.training_data.data)
            print("prev size: " + str(prev_size))
            size = len(self.training_data.data) // 16
            while size >= 1:
                size = math.ceil(size/2)
                if size > len(self.training_data.data):
                    size = 1
                print("size=" + str(size))
                training_data_copy = self.training_data.copy()
                training_data_copy.shuffle()
                batches = [training_data_copy.data[k:k + size] for k in range(0, len(training_data_copy.data), size)]
                added_back = 0
                examples_removed = 0
                for batch_i, batch in enumerate(batches):
                    print("[" + str(size) + "]" + "batch: " + str(batch_i) + "/" + str(len(batches)))
                    prev_accuracy = self.get_validation_accuracy()
                    removed_examples = []
                    for example in batch:
                        self.training_data.data.remove(example)
                        if util.get_highest_class(self.run(example)) != example[self.training_data.class_col]:
                            self.training_data.data.append(example)
                        else:
                            removed_examples.append(example)
                    if prev_accuracy > self.get_validation_accuracy() + .001:
                        print("- 0 removed")
                        added_back += 1
                        for example in removed_examples:
                            self.training_data.data.append(example)
                    else:
                        examples_removed += len(removed_examples)
                        print(" - " + str(len(removed_examples)) + " removed")
                print(str(added_back) + "/" + str(len(batches)) + " added back!")
                print(str(examples_removed) + " examples removed!")
                if size <= 8:
                    break
            print("new size: " + str(len(self.training_data.data)))










