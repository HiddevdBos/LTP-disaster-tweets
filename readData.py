import sys
import numpy as np
import torch


#From Exercise 3
def get_index(word, word2idx, freeze=False):
    """
    map words to indices
    keep special OOV token (_UNK) at position 0
    """
    if word in word2idx:
        return word2idx[word]
    else:
        if not freeze:
            word2idx[word]=len(word2idx) #new index
            return word2idx[word]
        else:
            return word2idx["_UNK"]

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
        data.append(item[-2:-1])
    return data, np.array(labels)

def transform_data(data):
    data_temp = []
    word2idx = {"_UNK": 0}  # reserve 0 for OOV
    for x in data:
        for sentence in x:
            sentence = sentence.split(" ")
            data_temp_sentence = []
            for w in sentence:
                data_temp_sentence.append(get_index(w,word2idx))
            data_temp.append(np.array(data_temp_sentence))
    return np.array(data_temp), len(word2idx)

#From Exercise 3
#TODO add max_length?
def convert_to_indices(X, max_length = 50):
    out = []
    for instance in X:
        indices = np.zeros(max_length, dtype=np.int)
        indices[:len(instance)] = instance
        indices[len(instance):] = 0
        out.append(indices)
    return np.array(out)

#From Exercise 3
def convert_to_n_hot(X, vocab_size):
    out = []
    for instance in X:
        n_hot = np.zeros(vocab_size)
        for w_idx in instance:
            n_hot[w_idx] = 1
        out.append(n_hot)
    return np.array(out)


# load data, change data to floats and get data to gpu (if cuda available)
def get_data(path):
    data, labels = load_data(path)
    data, vocab_size = transform_data(data)
    data = convert_to_n_hot(data, vocab_size)
    # data = convert_to_indices(data)
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels)
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    return data, labels
