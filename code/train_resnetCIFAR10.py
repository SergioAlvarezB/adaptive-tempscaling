import os
import time

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from utils import NumpyDataset


CIFAR10_PATH = '../data/CIFAR10'
SAVE_PATH = '../trained_models/CIFAR10/'
BATCH_SIZE = 128
MODEL = 'resnet50'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

if not os.path.exists(os.path.join(SAVE_PATH, MODEL)):
    os.mkdir(os.path.join(SAVE_PATH, MODEL))

dev = torch.device("cuda:0") if torch.cuda.is_available() \
    else torch.device('cpu')

def train_model(train_dataloader, val_dataloader, lrs=[1e-3, 1e-2, 1e-1], epochs=100):

    best_val = 0.00

    for lr in lrs:
        for weight_decay in np.append(np.geomspace(lr*1e-4, lr*1e-1, 4), [0.0]):
            print('Start new configuration:')
            print('\t Learning rate: {:.2e}'.format(lr))
            print('\t Weight Decay: {:.2e}'.format(weight_decay))

            loss = []
            acc = []
            val_loss = []
            val_acc = []


            # net = torchmodels.resnet50()
            # net.fc = nn.Linear(2048 , 2)
            
            # net = torchmodels.resnet18()
            # net.fc = nn.Linear(512 , 2)
            if MODEL == 'resnet50':
                net = resnet50()
                net.fc = torch.nn.Linear(2048 , 10)

            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr, momentum=0.9,
                                        nesterov=True,
                                        weight_decay=weight_decay)
            CE = torch.nn.functional.cross_entropy

            lr_sched = CosineAnnealingLR(optimizer, T_max=(len(train_dataloader)*epochs)//BATCH_SIZE)

            t = time.time()

            net.to(dev)

            for e in range(epochs):
                loss.append(0)
                acc.append(0)
                val_loss.append(0)
                val_acc.append(0)
                net.train()
                for x, y in train_dataloader:
                    x, y = x.to(dev), y.to(dev)

                    n = x.shape[0]

                    preds = net(x)
                    _loss = CE(preds, y, reduction='sum')
                    loss[-1] += _loss.item()
                    _, _preds = torch.max(preds, dim=1)
                    acc[-1] += torch.sum((_preds == y).float()).item() * 100.
                    
                    optimizer.zero_grad()
                    _loss.backward()
                    optimizer.step()
                    lr_sched.step()
                
                acc[-1] /= 40000 # Size of training set
                loss[-1] /= 40000
                    
                # Validation performance
                net.eval()
                for x, y in val_dataloader:
                    x = x.to(dev)

                    preds = net(x).detach().cpu()
                    _loss = CE(preds, y, reduction='sum')
                    val_loss[-1] += _loss.item()
                    _, _preds = torch.max(preds, dim=1)
                    val_acc[-1] += torch.sum((_preds == y).float()).item() * 100.
                
                val_acc[-1] /= 10000 # Size of validation set
                val_loss[-1] /= 10000
                    
                print("Time: {:.3f}s; train loss: {:.3f}, validation loss: {:.3f}".format(time.time() - t, loss[-1], val_loss[-1]))
                print("\t train accuracy: {:.2f}%, validation accuracy: {:.2f}%".format(acc[-1], val_acc[-1]))

            if val_acc[-1] > best_val:
                print('Found best configuration:')
                print('\t Learning rate: {:.2e}'.format(lr))
                print('\t Weight Decay: {:.2e}'.format(weight_decay))
                best_model = net
                best_val = val_acc[-1]

    return best_model

def predict_logits(model, dataloader):
    
    logits = []
    for x, _ in dataloader:
        logits.append(model(x).detach().numpy())
     
    return np.vstack(logits)


cifar10_transforms_train=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]) #transforms are different for train and test
cifar10_transforms_test=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

## Load Data
X_train = np.load(os.path.join(CIFAR10_PATH, 'train_imas.npy'))
y_train = np.load(os.path.join(CIFAR10_PATH, 'train_labels.npy'))

X_val = np.load(os.path.join(CIFAR10_PATH, 'val_imas.npy'))
y_val = np.load(os.path.join(CIFAR10_PATH, 'val_labels.npy'))

X_test = np.load(os.path.join(CIFAR10_PATH, 'test_imas.npy'))
y_test = np.load(os.path.join(CIFAR10_PATH, 'test_labels.npy'))


train_data = NumpyDataset(X_train, y_train, transform=cifar10_transforms_train)
val_data = NumpyDataset(X_val, y_val, transform=cifar10_transforms_test)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)


net = train_model(train_dataloader, val_dataloader)
model_path = os.path.join(SAVE_PATH)
torch.save(net, os.path.join(SAVE_PATH, MODEL))


# Precompute output-logits
test_data = NumpyDataset(X_test, y_test, transform=cifar10_transforms_test)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

train_data_test = NumpyDataset(X_train, y_train, transform=cifar10_transforms_test)
train_dataloader_test = DataLoader(train_data_test, batch_size=BATCH_SIZE)
train_logits = predict_logits(net, train_dataloader_test)
val_logits = predict_logits(net, val_dataloader)
test_logits = predict_logits(net, test_dataloader)

print('Accuracy on the different sets:')
print('\t Train set: {:.2f}'.format(100.0 * np.mean(np.argmax(train_logits, axis=1) == y_train)))
print('\t Val set: {:.2f}'.format(100.0 * np.mean(np.argmax(val_logits, axis=1) == y_val)))
print('\t Test set: {:.2f}'.format(100.0 * np.mean(np.argmax(test_logits, axis=1) == y_test)))


np.save(os.path.join(SAVE_PATH, MODEL, 'train_logits'), train_logits)
np.save(os.path.join(SAVE_PATH, MODEL, 'val_logits'), val_logits)
np.save(os.path.join(SAVE_PATH, MODEL, 'test_logits'), test_logits)

