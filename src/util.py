# util.py
# These are general utility functions that are useful across all of our algorithms/code.
import csv
import operator as op


# Calculates the class distribution of a 2D list of data. The distribution is stored in a dictionary that maps each
# class to the proportion of examples in 'data' that have that class.
def calculate_class_distribution(data, class_col):
    n = len(data)
    # This is our map of each class to its probability/proportion:
    probs = {}
    for obs in data:
        class_val = obs[class_col]
        # We either update the probabilities if the class was already present, or initialize the probability if not.
        # Note that we divide by n here in order to prevent having to do it in a future iteration of the probability
        # map.
        if class_val in probs:
            probs[class_val] += 1 / n
        else:
            probs[class_val] = 1 / n
    return probs

# This function takes in a probability distribution, outputs the class corresponding to the maximum probability. This
# would essentially return our "guess" for the class of an observation.
def get_highest_class(classes:dict) -> str:
    # We find the maximum by searching for the max probability
    return max(classes.items(), key=op.itemgetter(1))[0]


# Creates a 2D list from a file.
def read_file(filename):
    with open(filename) as csvfile:
        data = list(csv.reader(csvfile))
    empty_removed = []
    for line in data:
        if line:
            empty_removed.append(tuple(line))
    return empty_removed


# Counts the frequency of each class in the 2D list of data. Returns a map of each class to a count of the number of
# times it appears in the data.
def count_frequency(data):
        freq = {}
        for item in data:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        return freq