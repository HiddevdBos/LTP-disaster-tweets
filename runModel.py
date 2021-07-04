import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv 

from NN import NN

# Trains a model given input data and labels
def train_model(train_x, train_y, network):

    # batch size can be lowered in case of e.g. running out of memory
    batch_size = train_x.shape[0]
    
    num_batches = int(train_x.shape[0] / batch_size)
    train_x = train_x[(train_x.shape[0] % batch_size):]
    train_y = train_y[(train_y.shape[0] % batch_size):]

    train_x = train_x.reshape(num_batches, -1, train_x.shape[1])
    train_y = train_y.reshape(-1, batch_size)

    model = NN(train_x.shape[2], 2)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # transfer model and loss function to gpu
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # each model is trained for 10 epochs
    for epoch in range(0, 10):
        for idx in range(num_batches):
            # forward step
            prediction = model.forward(train_x[idx], network)
            loss = criterion(prediction, train_y[idx])

            # backwards step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # return the model and the average training loss
    return model

# evaluate performance of the model
def evaluate_model(model, data, labels, network):
    output = model.forward(data, network)
    total = labels.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == labels).sum()
    return correct / total

# main loop, performing k-fold cross validation, so training model k times on k sets of data
def cross_validation(data, labels, k, network):
    # divide data in k folds
    folds_x = list(torch.chunk(data, k))
    folds_y = list(torch.chunk(labels, k))
    acc_train_mean = 0
    acc_valid_mean = 0
    #k times choose another fold as validation set
    for fold in tqdm(range(0, k), desc='folds'):
        train_x = folds_x.copy()
        train_y = folds_y.copy()
        valid_x = train_x.pop(fold)
        valid_y = train_y.pop(fold)
        train_x = torch.cat(train_x)
        train_y = torch.cat(train_y)

        # train and evaluate the model with the given data
        model = train_model(train_x, train_y, network)
        acc_train = evaluate_model(model, train_x, train_y, network)
        acc_valid = evaluate_model(model, valid_x, valid_y, network)
        
        acc_valid_mean = (acc_valid_mean + acc_valid)
        acc_train_mean = (acc_train_mean + acc_train)

    acc_train_mean = acc_train_mean / k
    acc_valid_mean = acc_valid_mean / k

    print("training accuracy = ", acc_train_mean.item(), "validation accuracy = ", acc_valid_mean.item())

    return model

# make prediction given training and test data, used for running the test data
def make_prediction(train_data, train_labels, test_data, network):
    model = train_model(train_data, train_labels, network)    
    output = model.forward(test_data, network)
    _, predicted = torch.max(output.data, 1)
    ids = pd.read_csv('data/test.csv', usecols=["id"]).values
    with open('output.csv', 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(['id', 'target'])
        for idx, prediction in enumerate(predicted):
            csvwriter.writerow([ids[idx][0], prediction.item()])