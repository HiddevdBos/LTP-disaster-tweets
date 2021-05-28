import sys
from readData import get_data


def show_error():
    print("Please pass the required arguments")
    sys.exit()


if __name__ == '__main__':
    train_images, train_labels = get_data('data/train.csv')
    # test_images, test_labels = get_test_data('data/test.csv')