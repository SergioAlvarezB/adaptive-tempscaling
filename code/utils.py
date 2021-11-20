import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softmax as torch_softmax
from scipy.special import softmax, log_softmax

CIFAR10C_CATEGORIES = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 
    'snow', 'spatter', 'speckle_noise', 'zoom_blur']


CIFAR10_mean = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_std = np.array([0.2023, 0.1994, 0.2010])


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


def onehot_encode(X, n_classes=None):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    N = X.shape[0]

    if n_classes is None:
        n_classes = X.max()+1

    onehot = np.zeros((N, n_classes))
    onehot[np.arange(N), X] = 1.

    return onehot


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def torch_entropy(X, from_logits=True):
    if from_logits:
        X = torch_softmax(X, dim=1)

    H = -torch.sum(X*torch.log(X), dim=1)

    return H


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


def load_precomputedlogits(dataset, model, data_path='../data', to_tensor=False):

    # Build path.
    name = '_'.join([model, dataset])
    path = os.path.join(data_path, 'precomputed', name)

    # Load logits and labels
    name = '_'.join([dataset, model])
    logits_train = np.load(os.path.join(path, name
                                        + '_logit_prediction_train.npy'))
    logits_val = np.load(os.path.join(path, name
                                      + '_logit_prediction_valid.npy'))
    logits_test = np.load(os.path.join(path, name
                                       + '_logit_prediction_test.npy'))

    true_train = np.load(os.path.join(path, name + '_true_train.npy'))
    true_val = np.load(os.path.join(path, name + '_true_valid.npy'))
    true_test = np.load(os.path.join(path, name + '_true_test.npy'))

    if to_tensor:
        train = (torch.as_tensor(logits_train, dtype=torch.float32),
                torch.as_tensor(true_train, dtype=torch.int64))
        validation = (torch.as_tensor(logits_val, dtype=torch.float32),
                    torch.as_tensor(true_val, dtype=torch.int64))
        test = (torch.as_tensor(logits_test, dtype=torch.float32),
                torch.as_tensor(true_test, dtype=torch.int64))
    else:
        train = (logits_train, true_train)
        validation = (logits_val, true_val)
        test = (logits_test, true_test)

    return train, validation, test



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

    len_cal = max(len(cal) for cal in predictions.keys())
    
    print('{:>{width}}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('Calibrator', 'Accuracy', 'ECE', 'MCE', 'Brier Score', 'NLL', width=len_cal+1))
    for calibrator, preds in predictions.items():

        acc, ece, bri, nll, mce = compute_metrics(
                preds,
                target,
                M=M,
                from_logits=from_logits
            )

        res[calibrator] = [100*acc, ece, mce, bri, nll]
        print('{:<{width}}  {:>12.2f}% {:>12.2f}% {:>12.2f}% {:>12.3e}  {:>12.3e}'.format(calibrator, *res[calibrator], width=len_cal+1))

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
    mce = compute_mce(probs, target, M=M)
    bri = compute_brier(probs, target)
    nll = compute_nll(preds, target, from_logits=from_logits)

    return acc, ece, bri, nll, mce


def compute_brier(probs, target):
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

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

    if probs.ndim>1:
        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)

    else:
        confs = probs
        preds = probs >= 0.5

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
    ece *= 100

    return ece


def compute_mce(probs, target, M=15):
    """"Computes MCE score as defined in https://arxiv.org/abs/1706.04599"""

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

    mce = 0

    for low, high in zip(lows, highs):

        ix = (low < confs) & (confs <= high)
        n = np.sum(ix)
        if n<1:
            continue

        curr_preds = preds[ix]
        curr_confs = confs[ix]
        curr_target = target[ix]

        curr_acc = np.mean(curr_preds == curr_target)

        mce = max(mce, np.abs(np.mean(curr_confs)-curr_acc))

    mce *= 100

    return mce


def compute_nll(preds, target, from_logits=True, normalize=True):
    """Computes the negative-log-likelihood of a categorical model."""

    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    if from_logits:
        logits = log_softmax(preds)
    else:
        logits = np.log(preds)

    if normalize:
        return -np.mean(logits[np.arange(logits.shape[0]), target])

    else:
        return -np.sum(logits[np.arange(logits.shape[0]), target])


def binByEntropy(X, M=15, mode='same', from_logits=True, normalize=True):

    if mode not in ['log', 'same']:
        raise ValueError('mode must be one of log, same')

    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32)

    Hs = torch_entropy(X, from_logits=from_logits)

    N, dim = X.shape

    minH = max(1e-10, Hs.min().item())
    maxH = np.log(dim)

    if normalize:
        maxH = 1
        Hs /= np.log(dim)

    if mode=='log':
        lims = np.geomspace(minH, maxH+1e-10, M+1)


    if mode=='same':

        lims = np.quantile(Hs.numpy(), np.linspace(0, 1, M+1))
        lims[-1] += 1e-10

    ixs = []
    ## Compute idxs entropy per bin
    for i, (low, high) in enumerate(zip(lims[:-1], lims[1:])):
        ix = (low<=Hs) & (Hs<high)
        
        ixs.append(ix.numpy())


    return ixs, lims