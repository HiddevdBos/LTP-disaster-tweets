import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from NN import NN

def train_model(train_x, train_y, network):
    
    batch_size = train_x.shape[0]
    num_batches = int(train_x.shape[0] / batch_size)
    
    train_x = train_x[(train_x.shape[0] % batch_size):]
    train_y = train_y[(train_y.shape[0] % batch_size):]

    train_x = train_x.reshape(num_batches, -1, train_x.shape[1])
    train_y = train_y.reshape(-1, batch_size)
    
    model = NN(train_x.shape[2], 2)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

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

def evaluate_model(model, data, labels, network):
    output = model.forward(data, network)
    total = labels.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == labels).sum()
    return correct / total

def cross_validation(data, labels, k, network):
    folds_x = list(torch.chunk(data, k))
    folds_y = list(torch.chunk(labels, k))
    acc_train_mean = 0
    acc_valid_mean = 0
    for fold in tqdm(range(0, k), desc='folds'):
        train_x = folds_x.copy()
        train_y = folds_y.copy()
        valid_x = train_x.pop(fold)
        valid_y = train_y.pop(fold)
        train_x = torch.cat(train_x)
        train_y = torch.cat(train_y)

        model = train_model (train_x, train_y, network)
        acc_train = evaluate_model(model, train_x, train_y, network)
        acc_valid = evaluate_model(model, valid_x, valid_y, network)
        
        acc_valid_mean = (acc_valid_mean + acc_valid)
        acc_train_mean = (acc_train_mean + acc_train)

    acc_train_mean = acc_train_mean / k
    acc_valid_mean = acc_valid_mean / k

    print("training accuracy = ", acc_train_mean.item(), "validation accuracy = ", acc_valid_mean.item())