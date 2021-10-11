import os
import shutil

import numpy as np

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


CIFAR10_PATH = '../data/CIFAR10'

transforms_data=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train = CIFAR10(CIFAR10_PATH, train=True, download=True, transform=transforms_data)
test = CIFAR10(CIFAR10_PATH, transform=transforms_data)




# Divide into validation and train splits
N = len(train)

np.random.seed(1)
rand_ix = np.random.permutation(N)

ix_train, ix_val = rand_ix[:40000], rand_ix[40000:]

train_data = data.Subset(train, ix_train)
val_data = data.Subset(train, ix_val)

train = data.DataLoader(train_data, batch_size=40000, shuffle=True)
val = data.DataLoader(val_data, batch_size=10000, shuffle=True)
test = data.DataLoader(test, batch_size=1000)

X_train, y_train = next(iter(train))
X_val, y_val = next(iter(val))
X_test, y_test = next(iter(test))

X_train, y_train = X_train.numpy(), y_train.numpy()
X_val, y_val = X_val.numpy(), y_val.numpy()
X_test, y_test = X_test.numpy(), y_test.numpy()


for arrs, name in zip([(X_train, y_train), (X_val, y_val), (X_test, y_test)],
                      ['train', 'val', 'test']):
    np.save(os.path.join(CIFAR10_PATH, name + '_imas.npy'), arrs[0])
    np.save(os.path.join(CIFAR10_PATH, name + '_labels.npy'), arrs[1])


os.remove(os.path.join(CIFAR10_PATH, 'cifar-10-python.tar.gz'))
shutil.rmtree(os.path.join(CIFAR10_PATH, 'cifar-10-batches-py'))
