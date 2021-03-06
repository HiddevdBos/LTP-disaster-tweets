import sys
import argparse
from readData import get_data, get_test_data
from runModel import cross_validation, make_prediction

parser = argparse.ArgumentParser()
parser.add_argument('--network', help="type of network", type=str, default="linear")
args = parser.parse_args()

if __name__ == '__main__':
    # used for training, get train and validation accuracy
    train_data, train_labels, word2idx = get_data('data/train.csv', args.network)
    model = cross_validation(train_data, train_labels, 5, args.network)
    
    # get test data and test accuracy
    test_data = get_test_data('data/test.csv', args.network, word2idx)
    make_prediction(train_data, train_labels, test_data, args.network)