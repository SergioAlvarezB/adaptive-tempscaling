import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax, log_softmax

CIFAR10C_CATEGORIES = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 
    'snow', 'spatter', 'speckle_noise', 'zoom_blur']


CIFAR10_mean = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_std = np.array([0.2023, 0.1994, 0.2010])


def check_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def predict_logits(model, dataloader, dev):
    model.eval()
    model.to(dev)
    logits = []
    for x, _ in dataloader:
        x = x.to(dev)
        logits.append(model(x).detach().cpu().numpy())
     
    return np.vstack(logits)


def get_CIFAR10_C(dir_path):

    cifar10c = {}
    for cat in CIFAR10C_CATEGORIES:
        imas = np.load(os.path.join(dir_path, cat + '.npy'))
        cifar10c[cat] = {}
        for severity in range(5):
            cifar10c[cat][severity+1] = imas[10000*severity:10000*(severity+1)]
    
    cifar10c['labels'] = np.load(os.path.join(dir_path, 'labels.npy'))

    return cifar10c


def load_model(model, dataset, model_path='../trained_models/'):
    try:
        net = torch.load(os.path.join(model_path, dataset.upper(), model+'.pth'))
    except Exception as e:
        raise e

    return net


class NumpyDataset(torch.utils.data.Dataset):
    """Class to create a Pytorch Dataset from Numpy data"""

    def __init__(self, X, Y, transform=None, target_transform=None):
        super(NumpyDataset, self).__init__()

        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):

        ima,target = self.X[index], self.Y[index]

        if self.transform is not None:
            ima = self.transform(ima)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return ima, target


def calib_split(X, y, conf_th=None, apply_softmax=True):
    """Divides data into (high/low confidence)x(correct/incorrect)"""
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    correct = np.argmax(X, axis=1)==y

    if conf_th is None:
        conf_th = np.mean(correct)

    if apply_softmax:
        X = softmax(X, axis=1)

    confs = np.max(X, axis=1)

    high_conf = confs > conf_th

    hc = high_conf & correct
    lc = (~high_conf) & correct
    hi = high_conf & (~correct)
    li = (~high_conf) & (~correct)

    return hc, lc, hi, li



def compare_results(predictions, target, M=15, from_logits=True):

    res = {}

    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    print('{:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('Calibrator', 'Accuracy', 'ECE', 'Brier Score', 'NLL'))
    for calibrator, preds in predictions.items():

        acc, ece, bri, nll = compute_metrics(
                preds,
                target,
                M=M,
                from_logits=from_logits
            )

        res[calibrator] = [100*acc, 100*ece, bri, nll]
        print('{:>12}  {:>12.2f}  {:>12.2f}  {:>12.3e}  {:>12.3e}'.format(calibrator, *res[calibrator]))

    return res


def compute_metrics(preds, target, M=15, from_logits=True):
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    if from_logits:
        probs = softmax(preds, axis=1)
    else:
        probs = preds

    _preds = np.argmax(preds, axis=1)
    acc = np.mean(_preds == target)
    ece = compute_ece(probs, target, M=M)
    bri = compute_brier(probs, target)
    nll = compute_nll(preds, target, from_logits=from_logits)

    return acc, ece, bri, nll


def compute_brier(probs, target):
    targets = np.zeros(probs.shape)
    targets[np.arange(probs.shape[0]), target] = 1
    return np.mean(np.sum((probs - targets)**2, axis=1))


def compute_ece(probs, target, M=15):
    """"Computes ECE score as defined in https://arxiv.org/abs/1706.04599"""

    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    N = probs.shape[0]

    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    # Generate intervals
    limits = np.linspace(0, 1, num=M+1)
    lows, highs = limits[:-1], limits[1:]

    ece = 0

    for low, high in zip(lows, highs):

        ix = (low < confs) & (confs <= high)
        n = np.sum(ix)
        if n<1:
            continue

        curr_preds = preds[ix]
        curr_confs = confs[ix]
        curr_target = target[ix]

        curr_acc = np.mean(curr_preds == curr_target)

        ece += n*np.abs(np.mean(curr_confs)-curr_acc)

    ece /= N

    return ece


def compute_nll(preds, target, from_logits=True):
    """Computes the negative-log-likelihood of a categorical model."""
    if from_logits:
        logits = log_softmax(preds)
    else:
        logits = np.log(preds)

    return -np.sum(logits[np.arange(logits.shape[0]), target])