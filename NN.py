import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        # self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=self.embedding_dim)
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, output_dim)

    def forward(self, x):
        # x = self.embedding(x)
        # x = x.view(-1, self.embedding_dim *self.input_dim)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x
