import sys
import argparse
from readData import get_data
from runModel import cross_validation

parser = argparse.ArgumentParser()
parser.add_argument('--network', help="type of network", type=str, default="linear")
args = parser.parse_args()


def show_error():
    print("Please pass the required arguments")
    sys.exit()


if __name__ == '__main__':
    train_data, train_labels = get_data('data/train.csv', args.network)
    cross_validation(train_data, train_labels, 5, args.network)
    # test_data, test_labels = get_test_data('data/test.csv')