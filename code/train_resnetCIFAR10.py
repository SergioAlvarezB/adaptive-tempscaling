import numpy as np
import torch

from torch.utils import data, DataLoader
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

CIFAR10_PATH = '../data/CIFAR10'
BATCH_SIZE = 128


cifar10_transforms_train=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]) #transforms are different for train and test
cifar10_transforms_test=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

## Load Data
X = np.load('../data/CIFAR10/train_imas.npy')
Y = np.load('../data/CIFAR10/train_labels.npy')

Xval = np.load('/home/sergio/datasets/CIFAR10/cifar10_val_imas.npy')
Yval = np.load('/home/sergio/datasets/CIFAR10/cifar10_val_labels.npy')

Xtest = np.load('/home/sergio/datasets/CIFAR10/cifar10_test_imas.npy')
Ytest = np.load('/home/sergio/datasets/CIFAR10/cifar10_test_labels.npy')

