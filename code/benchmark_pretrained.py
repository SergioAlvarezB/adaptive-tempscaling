import os
import time


import torch
import numpy as np
import pandas as pd
from torch.nn.functional import softmax as torch_softmax
from code.models import HnLinearT

from models import NanValues, TempScaling, AdaTS, LinearT, DNNbasedT, HlogbasedT, BTS, EnsembleTS
from utils import compute_metrics, check_path, load_precomputedlogits, onehot_encode
from adats_utils import fitCV_AdaTS, fitAdaTS
from mixNmatch_cal import ets_calibrate, mir_calibrate


res_path = '../results/pretrained/'

check_path(res_path)


datasets = [
    'cifar10',
    'cifar100',
    'cars',
    'birds',
    'svhn'
]
lrs = [1e-4, 1e-5, 1e-5, 1e-5, 1e-4]
epochs = [80000, 100000, 160000, 160000, 80000]
models = [
    'densenet-121',
    'densenet-169',
    'lenet-5',
    'resnet-18',
    'resnet-50',
    'resnet-101',
    'resnext-29_8x16',
    'vgg-19',
    'wide-resnet-28x10',
    'wide-resnet-40x10'
]

TSmodels = [
    'TS',
    'ETS',
    'MIR',
    'BTS',
    'PTS',
    'LinearTS',
    'HTS',
    'HnLinearTS']

res_nll = pd.DataFrame(columns=['Dataset', 'Model', 'Uncalibrated'] + TSmodels)
res_ECE = pd.DataFrame(columns=['Dataset', 'Model', 'Uncalibrated'] + TSmodels)
res_bri = pd.DataFrame(columns=['Dataset', 'Model', 'Uncalibrated'] + TSmodels)

t0 = time.time()
ix = 0
for model in models:
    for _epochs, lr, dataset in zip(epochs, lrs, datasets):
        try:
            train, validation, test = load_precomputedlogits(dataset=dataset,
                                                             model=model,
                                                             data_path='../data')
        except:
            print('Configuration {}_{} not found.\n'.format(model, dataset))
            continue

        print('Start benchmarking for configuration {}_{} at time {:.1f}s\n'.format(model, dataset, time.time()-t0))

        X_train, Y_train = train
        X_val, Y_val = validation
        X_test, Y_test = test

        N, dim = X_train.shape

        ### Temp-Scal as baseline:
        tempScaler = TempScaling()
        tempScaler.fit(X_val, Y_val)

        TSmodels_predictive = {'TS': tempScaler.predictive}

        #### Mix-n-Match Baselines
        TSmodels_predictive['ETS'] = lambda x: ets_calibrate(X_val, onehot_encode(Y_val), x, dim)
        TSmodels_predictive['MIR'] = lambda x: mir_calibrate(X_val, onehot_encode(Y_val), x)

        #### BTS baseline
        bts = BTS()
        bts.fit(X_val, Y_val)
        TSmodels_predictive['BTS'] = bts.predictive

        ##### PTS baseline
        print('\tFitting PTS...')
        pts = AdaTS(DNNbasedT(dim, hs=[5, 5]))
        pts = fitAdaTS(pts, X_val, Y_val, epochs=_epochs, batch_size=1000, lr=lr, v=True)
        TSmodels_predictive['PTS'] = pts.predictive

        #### Our Models
        lts = AdaTS(LinearT(dim, norm=False))
        lts = fitAdaTS(lts, X_val, Y_val, epochs=_epochs, batch_size=1000, lr=1e-3, v=True)
        TSmodels_predictive['LinearTS'] = lts.predictive

        hts = AdaTS(HlogbasedT(dim))
        hts = fitAdaTS(hts, X_val, Y_val, epochs=_epochs, batch_size=1000, lr=1e-3, v=True)
        TSmodels_predictive['HTS'] = hts.predictive

        hnlts = AdaTS(HnLinearT(dim))
        hnlts = fitAdaTS(hnlts, X_val, Y_val, epochs=_epochs, batch_size=1000, lr=1e-4, v=True)
        TSmodels_predictive['HnLinearTS'] = hnlts.predictive



        acc, ece, bri, nll = compute_metrics(X_test, Y_test, M=15)

        eces = []
        bris = []
        nlls = []

        for label, TSmodel in TSmodels_predictive.items():
            TSmetrics = compute_metrics(TSmodel(X_test),
                                        Y_test,
                                        M=50,
                                        from_logits=False)

            eces.append(TSmetrics[1])
            bris.append(TSmetrics[2])
            nlls.append(TSmetrics[3])



        res_ECE.loc[ix] = [dataset, model, ece] + eces
        res_nll.loc[ix] = [dataset, model, nll] + nlls
        res_bri.loc[ix] = [dataset, model, bri] + bris

        ix += 1
        

res_ECE.to_csv('../results/pretrained/logitsJuan_ECE_reg.csv')
res_nll.to_csv('../results/pretrained/logitsJuan_NLL_reg.csv')
res_bri.to_csv('../results/pretrained/logitsJuan_BRI_reg.csv')
