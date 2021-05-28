import sys
import numpy as np
import torch


# load data and labels from csv file, store in lists
def load_data(path):
    data_file = open(path)
    data_list = []
    data = []
    labels = []
    
    for idx, line in enumerate(data_file):
        if idx == 0:
            continue
        line = str(line)[:-1]
        line = line.split(',')
        if line[0].isdigit():
            data_list.append(line)
        else:
            data_list[-1] += line

    for item in data_list:
        labels.append(int(item[-1]))
        data.append(item[:-1])

    return data, labels


# load data, change data to floats and get data to gpu (if cuda available)
def get_data(path):
    data, labels = load_data(path)
    # data = torch.from_numpy(data)
    # labels = torch.from_numpy(labels)
    # data = data.float()
    # if torch.cuda.is_available():
    #     data = data.cuda()
    #     labels = labels.cuda()
    return data, labels
