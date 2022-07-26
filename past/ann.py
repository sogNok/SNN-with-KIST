#Importing Library

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from bindsnet.encoding import Encoder, NullEncoder 

#Define Neural Networks Model.

class ECG(VisionDataset):
    training_file = './training_t1rc6.pt'
    test_file = './test_t1rc6.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five']
    
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(ECG, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            pass

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        #self.data = list(self.data)
        #self.targets = list(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        #if self.transform is not None:
        #    img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))


class TorchvisionDatasetWrapper(ECG):
    def __init__(
        self,
        image_encoder: Optional[Encoder] = None,
        label_encoder: Optional[Encoder] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.args = args
        self.kwargs = kwargs

        # Allow the passthrough of None, but change to NullEncoder
        if image_encoder is None:
            image_encoder = NullEncoder()

        if label_encoder is None:
            label_encoder = NullEncoder()

        self.image_encoder = image_encoder
        self.label_encoder = label_encoder

    def __getitem__(self, ind: int) -> Dict[str, torch.Tensor]:
        image, label = super().__getitem__(ind)
        output = (image.view(1, -1),label)

        return output

    def __len__(self):
        return super().__len__()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 6)

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 1280)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return F.log_softmax(h6, dim=1)

print("init model done")

# Set Hyper parameters and other variables to train the model.

batch_size = 4
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
no_cuda = False
seed = 1
log_interval = 200

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

n_workers = 4 * torch.cuda.device_count()
kwargs = {'num_workers': n_workers, 'pin_memory': True} if use_cuda else {}

print("set vars and device done")

# Load MNIST data.
train_dataset = TorchvisionDatasetWrapper(
    None,    #PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "ECG"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)

# Load MNIST data.
test_dataset = TorchvisionDatasetWrapper(
    None, #PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "ECG"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor()]#, transforms.Lambda(lambda x: x * intensity)]
    ),
)

#Prepare Data Loader for Training and Validation
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#Define Train function and Test function to validate.

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(log_interval, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Train and Test the model and save it.

for epoch in range(1, 11):
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)
torch.save(model, './model.pt')

