import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import os
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

class ECG(data.Dataset):
    training_file = './training_t1rc6.pt'
    test_file = './test_t1rc6.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five']
    
    def __init__(
            self,
            root: str,
            train: bool = True,
    ) -> None:
        super(ECG, self).__init__()
        self.train = train  # training set or test set
        self.root = root

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, data_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        datum, target = self.data[index], int(self.targets[index])

        return datum, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.root,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.root,
                                            self.test_file)))

class Outlayer(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(Outlayer, self).__init__()

        self.data = x_tensor
        self.targets = y_tensor

    def __getitem__(self, index: int) -> Tuple[Any,Any]:
        datum, target = self.data[index], int(self.targets[index])

        return datum, target
    def __len__(self):
        return len(self.data)

class CNN(torch.nn.Module):

    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.keep_prob = 0.5

        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = nn.Linear(self.input_size // 8 * 128, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = nn.Linear(625, self.num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        h = int(input_size/2)
        #self.linear_1 = nn.Linear(input_size, num_classes)
        self.linear_1 = nn.Linear(input_size, h)
        self.linear_2 = nn.Linear(h, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        out = torch.sigmoid(self.linear_2(out))
        return out

class ANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(1,-1)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return F.log_softmax(h6, dim=1)
