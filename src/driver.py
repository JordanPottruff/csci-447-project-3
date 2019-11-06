# driver.py
# File for running our experimental design.
import src.data.data as d
import src.mfnn as mfnn

def main():
    test_data = d.get_test_data().get_data()
    training_set = test_data
    validation_set = test_data * 2

    multilayer = mfnn.MFNN(training_set, validation_set, [2, 2, 2], None, None, None)
    example = test_data[0][:-1]
    multilayer.backpropagation(example, [1,0])


main()
