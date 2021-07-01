import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, num_features=3):
        super().__init__()
        self.num_features = num_features
        self.conv1 = nn.Conv1d(num_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_features*num_features)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input.shape == (B,N,3)
        input = input.transpose(1, 2)
        batch_size = input.size(0)
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        pool = nn.MaxPool1d(x.size(-1))(x)
        flat = nn.Flatten(1)(pool)
        x = F.relu(self.bn4(self.fc1(flat)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        matrix = x.view(-1,self.num_features,self.num_features)

        # Initialize bias as identity matrix
        bias = torch.eye(self.num_features,requires_grad=True).repeat(batch_size,1,1)
        if matrix.is_cuda:
            bias = bias.cuda()
        matrix += bias
        return matrix


class PointNet(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()
        self.input_transform = TNet(num_features=3)
        self.featrue_transform = TNet(num_features=64)

        self.conv1_1 = nn.Conv1d(3,64,1)
        self.conv1_2 = nn.Conv1d(64,64,1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2_1 = nn.Conv1d(64,64,1)
        self.conv2_2 = nn.Conv1d(64,128,1)
        self.conv2_3 = nn.Conv1d(128,1024,1)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(1024)

        self.clf = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256,num_classes),
        )

    def forward(self,input):
        # input.shape==(B,N,3)
        matrix_3x3 = self.input_transform(input) 
        x = torch.bmm(input,matrix_3x3).transpose(1,2) # (B,N,3) -> (B,3,N)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))

        # x.shape == (B,64,N)
        matrix_64x64 = self.featrue_transform(x.transpose(1,2))
        x = torch.bmm(torch.transpose(x,1,2),matrix_64x64).transpose(1,2) # (B,N,64) -> (B,64,N)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.bn2_3(self.conv2_3(x))

        # Global Maxpooling,x.shape == (B,1024,N)
        x = nn.MaxPool1d(x.size(-1))(x)
        features = nn.Flatten(1)(x)
        
        # classification
        output = self.clf(features)
        return output,matrix_3x3,matrix_64x64
