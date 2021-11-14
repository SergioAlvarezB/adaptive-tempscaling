import os
import time


import torch
import numpy as np
import pandas as pd
from torch.nn.functional import softmax as torch_softmax

from models import NanValues, TempScaling, AdaTS, ScaleT, LinearT, HbasedT, DNNbasedT, HlogbasedT, HistTS
from utils import compute_metrics, check_path, load_precomputedlogits
from adats_utils import fitCV_AdaTS, fitHistTS


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
    'HisTS'
    'ScaleT',
    'LinearTS',
    'HTS',
    'lHTS',
    'DNNTs']

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

        ### Histogram based
        hisTS = fitHistTS(X_val, Y_val,
                        Ms=[5, 10, 15],
                        iters=3)

        tempScalerModels = {'TS': tempScaler,
                            'HisTS': hisTS}

        wds = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0])

        for TSmodel, TSclass in zip(TSmodels[2:],
                                    [ScaleT, LinearT, HbasedT, HlogbasedT, DNNbasedT]):

            print('Fitting TS model: {}'.format(TSmodel))
            target_file = os.path.join(res_path, '{}_{}_{}.png'.format(dataset, model, TSmodel))
            failed=True
            while failed:
                try:
                    tempScalerModels[TSmodel] = fitCV_AdaTS(TSclass,
                        X_val,
                        Y_val,
                        epochs=_epochs,
                        batch_size=1000,
                        lrs=lr,
                        v=True,
                        weight_decays=(wds*dim) if TSmodel in ['ScaleT', 'LinearTS', 'DNNTs'] else 0.,
                        target_file=target_file,
                        iters=3)
                    failed = False
                except NanValues:
                    lr*=0.1
                print('\n')

        acc, ece, bri, nll = compute_metrics(X_test, Y_test, M=15)

        eces = []
        bris = []
        nlls = []

        for label, TSmodel in tempScalerModels.items():
            TSmetrics = compute_metrics(TSmodel.predictive(X_test),
                                        Y_test,
                                        M=15,
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
