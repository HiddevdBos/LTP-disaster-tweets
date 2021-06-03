import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        
        self.vocab_size = 30938

        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = 200)

        self.lstm = nn.LSTM(200, 100)
        self.lstm2 = nn.LSTM(100, 50)

        self.fc1 = nn.Linear(50 * 50, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, output_dim)

        self.fc1_linear = nn.Linear(input_dim, 1500)
        self.fc2_linear = nn.Linear(1500, 40)
        # self.fc3_linear = nn.Linear(200 , 15)
        self.fc4_linear = nn.Linear(40 ,output_dim)

    def forward(self, x, network = "embedding"):
        if network == "embedding":
            
            x = self.embedding(F.relu(x))

            x,_ = self.lstm(F.relu(x))
            x,_ = self.lstm2(F.relu(x))
            # x = x[:,-1,:]
            x = x.view(-1, 50*50)

            x = self.fc1(F.relu(x))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))
            # x = F.log_softmax(x, dim=1)
        
        if network == "linear":

            x = self.fc1_linear(F.relu(x))
            x = self.fc2_linear(F.relu(x))
            # x = self.fc3_linear(F.relu(x))
            x = self.fc4_linear(F.relu(x))

        return x
