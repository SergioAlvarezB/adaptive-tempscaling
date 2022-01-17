import os
import time

import pandas as pd
import numpy as np
from utils import softmax

from models import AdaTS, PTS, NanValues
from models import LTS as LTS_torch
from models import HTS as HTS_torch
from models import HnLTS as HnLTS_torch
from scipy_models import TS, LTS, HTS, HnLTS, BTS
from utils import compute_metrics, check_path, load_precomputedlogits, onehot_encode, compute_ece
from adats_utils import fitAdaTS
from mixNmatch_cal import ets_calibrate, mir_calibrate


res_path = '../results/pretrained/'

check_path(res_path)

for i in range(50):
    datasets = [
        'cifar10',
        'cifar100',
    ]
    lrs = [1e-3, 1e-3]
    epochs = [800000, 1000000]
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
        'PTS_ece',
        'LTS',
        'HTS',
        'HnLTS',
        'LTS_torch',
        'HTS_torch',
        'HnLTS_torch']

    Ns = [200, 1000]

    res_nll = pd.DataFrame(columns=['N', 'Dataset', 'Model', 'Uncalibrated'] + TSmodels)
    res_ECE = pd.DataFrame(columns=['N', 'Dataset', 'Model', 'Uncalibrated'] + TSmodels)
    res_ECE15 = pd.DataFrame(columns=['N', 'Dataset', 'Model', 'Uncalibrated'] + TSmodels)
    res_bri = pd.DataFrame(columns=['N', 'Dataset', 'Model', 'Uncalibrated'] + TSmodels)

    t0 = time.time()
    ix = 0
    for model in models:
        for _epochs, lr, dataset in zip(epochs, lrs, datasets):
            for N in Ns:
                try:
                    train, validation, test = load_precomputedlogits(dataset=dataset,
                                                                    model=model,
                                                                    data_path='../data')
                except Exception as e:
                    print(e)
                    print('Configuration {}_{} not found.\n'.format(model, dataset))
                    continue

                print('Start benchmarking for configuration {}_{} at time {:.1f}s\n'.format(model, dataset, time.time()-t0))

                X_train, Y_train = train
                X_val, Y_val = validation
                X_test, Y_test = test

                ### Data Subsampling
                ix_rand = np.random.permutation(X_val.shape[0])[:N]
                X_val, Y_val = X_val[ix_rand], Y_val[ix_rand]

                _, dim = X_train.shape

                ### Temp-Scal as baseline:
                tempScaler = TS(dim)
                tempScaler.fit(X_val, Y_val)

                TSmodels_predictive = {'TS': tempScaler.predictive}

                #### Mix-n-Match Baselines
                TSmodels_predictive['ETS'] = lambda x: ets_calibrate(X_val, onehot_encode(Y_val, n_classes=dim), x, dim)
                TSmodels_predictive['MIR'] = lambda x: mir_calibrate(X_val, onehot_encode(Y_val, n_classes=dim), x)

                #### BTS baseline
                print('\tFitting BTS...')
                bts = BTS()
                bts.fit(X_val, Y_val, v=True)
                TSmodels_predictive['BTS'] = bts.predictive

                ##### PTS baseline
                print('\n\tFitting PTS...')
                failed = True
                curr_epochs = _epochs
                curr_lr = lr
                while failed:
                    try:
                        pts = AdaTS(PTS(dim))
                        pts = fitAdaTS(pts, X_val, Y_val,
                                    target_file='../results/pretrained/PTS_{}_{}_{:d}.png'.format(model, dataset, i),
                                    epochs=curr_epochs, 
                                    batch_size=1000,
                                    lr=curr_lr,
                                    v=True,
                                    optimizer='sgd')
                        failed = False
                    except Exception as e:
                        if type(e) == NanValues:
                            curr_epochs *= 2
                            curr_lr /=2

                TSmodels_predictive['PTS'] = pts.predictive


                ##### PTS-ECE baseline
                print('\n\tFitting PTS_ece...')
                failed = True
                curr_epochs = _epochs
                curr_lr = lr/10
                while failed:
                    try:
                        pts_ece = AdaTS(PTS(dim))
                        pts_ece = fitAdaTS(pts_ece, X_val, Y_val,
                                        target_file='../results/pretrained/PTSece_{}_{}_{:d}.png'.format(model, dataset, i),
                                        loss='ece',
                                        epochs=curr_epochs,
                                        batch_size=10000,
                                        lr=curr_lr,
                                        v=True,
                                        optimizer='sgd')
                        failed = False
                    except Exception as e:
                        if type(e) == NanValues:
                            curr_epochs *= 2
                            curr_lr /=2

                TSmodels_predictive['PTS_ece'] = pts_ece.predictive

                #### Our Models
                print('\n\tFitting LTS...')
                lts = LTS(dim)
                lts.fit(X_val, Y_val, v=True)
                TSmodels_predictive['LTS'] = lts.predictive

                print('\n\tFitting LTS_torch...')
                failed = True
                curr_epochs = _epochs
                curr_lr = lr
                while failed:
                    try:
                        lts_t = AdaTS(LTS_torch(dim))
                        lts_t = fitAdaTS(lts_t, X_val, Y_val,
                                        target_file='../results/pretrained/LTS_{}_{}_{:d}.png'.format(model, dataset, i),
                                        epochs=curr_epochs,
                                        batch_size=1000,
                                        lr=curr_lr,
                                        v=True,
                                        optimizer='sgd')
                        failed = False
                    except Exception as e:
                        if type(e) == NanValues:
                            curr_epochs *= 2
                            curr_lr /=2
                TSmodels_predictive['LTS_torch'] = lts_t.predictive


                print('\n\tFitting HTS...')
                hts = HTS(dim)
                hts.fit(X_val, Y_val, v=True)
                TSmodels_predictive['HTS'] = hts.predictive

                print('\n\tFitting HTS_torch...')
                failed = True
                curr_epochs = _epochs
                curr_lr = lr
                while failed:
                    try:
                        hts_t = AdaTS(HTS_torch(dim))
                        hts_t = fitAdaTS(hts_t, X_val, Y_val,
                                        target_file='../results/pretrained/HTS_{}_{}_{:d}.png'.format(model, dataset, i),
                                        epochs=curr_epochs,
                                        batch_size=1000,
                                        lr=curr_lr,
                                        v=True,
                                        optimizer='sgd')
                        failed = False
                    except Exception as e:
                        if type(e) == NanValues:
                            print('Restart training due to Nan values... \n')
                            curr_epochs *= 2
                            curr_lr /=2
                        raise e
                TSmodels_predictive['HTS_torch'] = hts_t.predictive

                
                

                print('\n\tFitting HnLTS...')
                hnlts = HnLTS(dim)
                hnlts.fit(X_val, Y_val, v=True)
                TSmodels_predictive['HnLTS'] = hnlts.predictive

                print('\n\tFitting HnLTS_torch...')
                failed = True
                curr_epochs = _epochs
                curr_lr = lr
                while failed:
                    try:
                        hnlts_t = AdaTS(HnLTS_torch(dim))
                        hnlts_t = fitAdaTS(hnlts_t, X_val, Y_val,
                                        target_file='../results/pretrained/HnLTS_{}_{}_{:d}.png'.format(model, dataset, i),
                                        epochs=curr_epochs,
                                        batch_size=1000,
                                        lr=curr_lr,
                                        v=True,
                                        optimizer='sgd')
                        failed = False
                    except Exception as e:
                        if type(e) == NanValues:
                            curr_epochs *= 2
                            curr_lr /=2
                TSmodels_predictive['HnLTS_torch'] = hnlts_t.predictive



                acc, ece, bri, nll, mce = compute_metrics(X_test, Y_test, M=50, from_logits=True)
                ece_15 = (compute_ece(softmax(X_test, axis=-1), Y_test))

                eces = []
                bris = []
                nlls = []
                eces_15 = []

                for label, TSmodel in TSmodels_predictive.items():
                    TSmetrics = compute_metrics(TSmodel(X_test),
                                                Y_test,
                                                M=50,
                                                from_logits=False)

                    eces_15.append(compute_ece(TSmodel(X_test), Y_test))
                    eces.append(TSmetrics[1])
                    bris.append(TSmetrics[2])
                    nlls.append(TSmetrics[3])


                res_ECE15.loc[ix] = [N, dataset, model, ece_15] + eces_15
                res_ECE.loc[ix] = [N, dataset, model, ece] + eces
                res_nll.loc[ix] = [N, dataset, model, nll] + nlls
                res_bri.loc[ix] = [N, dataset, model, bri] + bris

                ix += 1
            

    res_ECE15.to_csv('../results/pretrained/logitsJuan_ECE15_red_{:d}.csv'.format(i))
    res_ECE.to_csv('../results/pretrained/logitsJuan_ECE_red_{:d}.csv'.format(i))
    res_nll.to_csv('../results/pretrained/logitsJuan_NLL_red_{:d}.csv'.format(i))
    res_bri.to_csv('../results/pretrained/logitsJuan_BRI_red_{:d}.csv'.format(i))
