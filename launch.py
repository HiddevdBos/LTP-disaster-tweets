import sys
from readData import get_data
from runModel import cross_validation

def show_error():
    print("Please pass the required arguments")
    sys.exit()


if __name__ == '__main__':
    train_data, train_labels = get_data('data/train.csv')
    cross_validation(train_data, train_labels, 5)
    # test_data, test_labels = get_test_data('data/test.csv')