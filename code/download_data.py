import os
import shutil

import numpy as np

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from utils import check_path


CIFAR10_PATH = '../data/CIFAR10'

check_path(CIFAR10_PATH)

transforms_data=transforms.Compose([
    transforms.ToTensor(),
])

train = CIFAR10(CIFAR10_PATH, train=True, download=True, transform=transforms_data)
test = CIFAR10(CIFAR10_PATH, transform=transforms_data)




# Divide into validation and train splits
N = len(train)

np.random.seed(10)
rand_ix = np.random.permutation(N)

ix_train, ix_val = rand_ix[:40000], rand_ix[40000:]

train_data = data.Subset(train, ix_train)
val_data = data.Subset(train, ix_val)

train = data.DataLoader(train_data, batch_size=40000)
val = data.DataLoader(val_data, batch_size=10000)
test = data.DataLoader(test, batch_size=1000)

X_train, y_train = next(iter(train))
X_val, y_val = next(iter(val))
X_test, y_test = next(iter(test))

X_train, y_train = (255*X_train.numpy()).astype(np.uint8), y_train.numpy()
X_val, y_val = (255*X_val.numpy()).astype(np.uint8), y_val.numpy()
X_test, y_test = (255*X_test.numpy()).astype(np.uint8), y_test.numpy()

X_train = np.transpose(X_train, (0, 2,3,1))
X_val= np.transpose(X_val, (0, 2,3,1))
X_test = np.transpose(X_test, (0, 2,3,1))

for arrs, name in zip([(X_train, y_train), (X_val, y_val), (X_test, y_test)],
                      ['train', 'val', 'test']):
    np.save(os.path.join(CIFAR10_PATH, name + '_imas.npy'), arrs[0])
    np.save(os.path.join(CIFAR10_PATH, name + '_labels.npy'), arrs[1])


os.remove(os.path.join(CIFAR10_PATH, 'cifar-10-python.tar.gz'))
shutil.rmtree(os.path.join(CIFAR10_PATH, 'cifar-10-batches-py'))
