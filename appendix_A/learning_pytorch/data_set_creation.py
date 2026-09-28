from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

x_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

x_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

# A class is a category or possible value that a model is trying to predict.
# For example, predicting digits 0-9 means there are 10 classes, one for each digit.

class ToyDataset(Dataset):
    def __init__(self, X, Y):
        self.features = X
        self.labels = Y

    def __getitem__(self, index) -> Any:
        '''
        Instructions for retrieving exactly one data record and the corresponding label
        '''
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0] # returns total length of the dataset


train_ds = ToyDataset(x_train, y_train)
test_ds  = ToyDataset(x_test, y_test)


torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:",x,y)