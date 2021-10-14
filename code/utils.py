from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax, log_softmax


def check_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

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


def compare_results(predictions, target, M=15, from_logits=True):

    res = {}

    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    print('{:>12}  {:>12}  {:>12}  {:>12}  {:>12}'.format('Calibrator', 'Accuracy', 'ECE', 'Brier Score', 'NLL'))
    for calibrator, preds in predictions.items():

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()

        if from_logits:
            probs = softmax(preds, axis=1)
        else:
            probs = preds

        _preds = np.max(preds, axis=1)
        acc = np.mean(_preds == target)
        ece = compute_ece(probs, target, M=M)
        bri = compute_brier(probs, target)
        nll = compute_nll(preds, target, from_logits=from_logits)

        res[calibrator] = [100*acc, 100*ece, bri, nll]
        print('{:>12}  {:>12.2f}  {:>12.2f}  {:>12.3e}  {:>12.3e}'.format(calibrator, *res[calibrator]))

    return res


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

    confs = probs[np.arange(probs.shape[0]), target]
    preds = np.max(probs, axis=1)

    # Generate intervals
    limits = np.linspace(0, 1, num=M+1)
    lows, highs = limits[:-1], limits[1:]

    ece = 0

    for low, high in zip(lows, highs):
        if not any((low < confs) & (confs <= high)):
            continue
        curr_preds = preds[(low < confs) & (confs <= high)]
        curr_confs = confs[(low < confs) & (confs <= high)]
        curr_target = target[(low < confs) & (confs <= high)]

        curr_acc = np.mean(curr_preds == curr_target)

        ece += curr_target.size*np.abs(np.mean(curr_confs)-curr_acc)

    ece /= probs.shape[0]

    return ece


def compute_nll(preds, target, from_logits=True):
    """Computes the negative-log-likelihood of a categorical model."""
    if from_logits:
        logits = log_softmax(preds)
    else:
        logits = np.log(preds)

    return -np.sum(logits[np.arange(logits.shape[0]), target])