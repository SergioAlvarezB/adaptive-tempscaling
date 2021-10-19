from ctypes import ArgumentError
import os
import time
import argparse

import numpy as np
import torch
from torch.cuda import init

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet50, resnet101, densenet121
import torchvision.transforms as transforms
from models import LeNet5

from utils import NumpyDataset, check_path


CIFAR10_PATH = '../data/CIFAR10'
SAVE_PATH = '../trained_models/CIFAR10/'
MODELS = ['densenet121', 'resnet101', 'resnet50', 'lenet5']

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='model to train', type=str.lower,
                        required=True)
    parser.add_argument('--dev', help='Device to use', type=str.lower, default='cpu')

    parser.add_argument('--batch_size', help='Mini-batch size to use for training',
                        type=int, default=128)
    parser.add_argument('--epochs', help='Number of epochs',
                        type=int, default=200)
    parser.add_argument('--lr', nargs='+', help='Learning rates to try',
                        default=[1e-4, 1e-3], type=float)
    parser.add_argument('-wd', '--weight_decay', nargs='+', help='Values of weight decay to use',
                        default=np.geomspace(1e-4, 1e-2, 3), type=float)
    parser.add_argument('--pretrained', help='If set loads pretrained model', action='store_true')

    return parser.parse_args()




def train_model(init_model,
                train_dataloader,
                val_dataloader,
                lrs=[1e-4, 1e-3],
                weight_decays=np.geomspace(1e-4, 1e-2, 3),
                epochs=200,
                dev=torch.device('cpu')):

    best_val = np.inf

    for lr in lrs:
        for weight_decay in weight_decays:
            print('Start new configuration:')
            print('\t Learning rate: {:.2e}'.format(lr))
            print('\t Weight Decay: {:.2e}'.format(weight_decay))

            loss = []
            acc = []

            net = init_model()
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr, momentum=0.9,
                                        nesterov=True,
                                        weight_decay=weight_decay)
            CE = torch.nn.functional.cross_entropy

            lr_sched = MultiStepLR(optimizer, milestones=[100, 150])

            t = time.time()

            net.to(dev)

            for e in range(epochs):
                loss.append(0)
                acc.append(0)
                net.train()
                N = 0
                for x, y in train_dataloader:
                    x, y = x.to(dev), y.to(dev)

                    n = x.shape[0]
                    N += n

                    preds = net(x)
                    _loss = CE(preds, y, reduction='mean')
                    loss[-1] += n*_loss.item()
                    _, _preds = torch.max(preds, dim=1)
                    acc[-1] += torch.sum((_preds == y).float()).item() * 100.
                    
                    optimizer.zero_grad()
                    _loss.backward()
                    optimizer.step()
                    
                
                acc[-1] /= N # Size of training set
                loss[-1] /= N
            
                if e <= int(0.8*epochs):
                    lr_sched.step()
                
                if e%5==1:  
                    print('End of epoch: {:d}'.format(e))
                    print("Time: {:.3f}s; train loss: {:.3f}".format(time.time() - t, loss[-1]))
                    print("\t train accuracy: {:.2f}%".format(acc[-1]))

            net.eval()
            val_loss = 0
            val_acc = 0
            Nval = 0
            for x, y in val_dataloader:
                x = x.to(dev)

                Nval += x.shape[0]

                preds = net(x).detach().cpu()
                _loss = CE(preds, y, reduction='sum')
                val_loss += _loss.item()
                _, _preds = torch.max(preds, dim=1)
                val_acc += torch.sum((_preds == y).float()).item() * 100.
            
            val_acc /= Nval # Size of validation set
            val_loss /= Nval
            print('End of epoch: {:d}'.format(e))
            print("Time: {:.3f}s; train loss: {:.3f}, validation loss: {:.3f}".format(time.time() - t, loss[-1], val_loss))
            print("\t train accuracy: {:.2f}%, validation accuracy: {:.2f}%".format(acc[-1], val_acc))

            if val_loss < best_val:
                print('Found best configuration:')
                print('\t Learning rate: {:.2e}'.format(lr))
                print('\t Weight Decay: {:.2e}'.format(weight_decay))
                best_model = net
                best_val = val_loss

    return best_model

def predict_logits(model, dataloader, dev):

    model.eval()
    model.to(dev)
    logits = []
    for x, _ in dataloader:
        x = x.to(dev)
        logits.append(model(x).detach().cpu().numpy())
     
    return np.vstack(logits)


def main():

    conf = parse_arguments()
    dev = torch.device(conf.dev) if torch.cuda.is_available() \
        else torch.device('cpu')

    if conf.model not in MODELS:
        raise ArgumentError('model not recognized, must be one of {}\n'.format(MODELS))

    check_path(os.path.join(SAVE_PATH, conf.model))

    if conf.model == 'densenet121':
        def init_model():
            net = densenet121()
            net.fc = torch.nn.Linear(1024, 10)
            return net

    if conf.model == 'resnet101':
        def init_model():
            net = resnet101()
            net.fc = torch.nn.Linear(2048 , 10)
            return net

    if conf.model == 'resnet50':
        def init_model():
            net = resnet50()
            net.fc = torch.nn.Linear(2048 , 10)
            return net

    elif conf.model == 'lenet5':
        def init_model():
            net = LeNet5(dim=10, input_channels=3)
            return net


    cifar10_transforms_train=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) #transforms are different for train and test
    cifar10_transforms_test=transforms.Compose([
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

    N = X_train.shape[0]
    Nval = X_val.shape[0]


    train_data = NumpyDataset(X_train, y_train, transform=cifar10_transforms_train)
    val_data = NumpyDataset(X_val, y_val, transform=cifar10_transforms_test)

    train_dataloader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=conf.batch_size)

    if not conf.pretrained:
        net = train_model(init_model,
                        train_dataloader,
                        val_dataloader,
                        lrs=conf.lr,
                        epochs=conf.epochs,
                        weight_decays=conf.weight_decay,
                        dev=dev)
        torch.save(net, os.path.join(SAVE_PATH, conf.model+'.pth'))
    else:
        try:
            net = torch.load(os.path.join(SAVE_PATH, conf.model+'.pth'))
        except Exception as e:
            raise e


    # Precompute output-logits
    test_data = NumpyDataset(X_test, y_test, transform=cifar10_transforms_test)
    test_dataloader = DataLoader(test_data, batch_size=conf.batch_size)

    train_data_test = NumpyDataset(X_train, y_train, transform=cifar10_transforms_test)
    train_dataloader_test = DataLoader(train_data_test, batch_size=conf.batch_size)
    train_logits = predict_logits(net, train_dataloader_test, dev=dev)
    val_logits = predict_logits(net, val_dataloader, dev=dev)
    test_logits = predict_logits(net, test_dataloader, dev=dev)

    print('Accuracy on the different sets:')
    print('\t Train set: {:.2f}'.format(100.0 * np.mean(np.argmax(train_logits, axis=1) == y_train)))
    print('\t Val set: {:.2f}'.format(100.0 * np.mean(np.argmax(val_logits, axis=1) == y_val)))
    print('\t Test set: {:.2f}'.format(100.0 * np.mean(np.argmax(test_logits, axis=1) == y_test)))


    np.save(os.path.join(SAVE_PATH, conf.model, 'train_logits'), train_logits)
    np.save(os.path.join(SAVE_PATH, conf.model, 'val_logits'), val_logits)
    np.save(os.path.join(SAVE_PATH, conf.model, 'test_logits'), test_logits)


if __name__ == '__main__':
    main()