import os
import time

import pandas as pd
import numpy as np
from utils import softmax

from models import AdaTS, PTS, NanValues
from models import LTS, HTS, HnLTS
from scipy_models import TS, BTS
from utils import compute_metrics, check_path, load_precomputedlogits, onehot_encode, compute_ece
from adats_utils import fitAdaTS
from mixNmatch_cal import ets_calibrate, mir_calibrate


res_path = '../results/data_eff/'

check_path(res_path)

dataset = 'cifar10'
Ns = np.geomspace(100, 10000, 10, dtype=np.int)
models = [
    'densenet-121',
    'vgg-19'
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
    'HnLTS']

for i in range(50):
    

    res_nll = pd.DataFrame(columns=['N', 'Model', 'Uncalibrated'] + TSmodels)
    res_ECE = pd.DataFrame(columns=['N', 'Model', 'Uncalibrated'] + TSmodels)
    res_ECE15 = pd.DataFrame(columns=['N', 'Model', 'Uncalibrated'] + TSmodels)
    res_bri = pd.DataFrame(columns=['N', 'Model', 'Uncalibrated'] + TSmodels)

    t0 = time.time()
    ix = 0
    for model in models:
        for N in Ns:
            try:
                train, validation, test = load_precomputedlogits(dataset=dataset,
                                                                model=model,
                                                                data_path='../data')
            except Exception as e:
                print(e)
                print('Configuration {}_{} not found.\n'.format(model, dataset))
                continue

            print('Start benchmarking for configuration {}_{} at time {:.1f}s\n'.format(model, N, time.time()-t0))

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
            TSmodels_predictive['ETS'] = lambda x: ets_calibrate(X_val, onehot_encode(Y_val), x, dim)
            TSmodels_predictive['MIR'] = lambda x: mir_calibrate(X_val, onehot_encode(Y_val), x)

            #### BTS baseline
            print('\tFitting BTS...')
            bts = BTS()
            bts.fit(X_val, Y_val, v=True)
            TSmodels_predictive['BTS'] = bts.predictive

            ##### PTS baseline
            print('\n\tFitting PTS...')
            failed = True
            curr_epochs = 100000
            curr_lr = 1e-3
            while failed:
                try:
                    pts = AdaTS(PTS(dim))
                    pts = fitAdaTS(pts, X_val, Y_val,
                                   target_file=os.path.join(res_path, 'PTS_{}_{}_{:d}.png'.format(model, N, i)),
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
            curr_epochs = 100000
            curr_lr = 1e-3
            while failed:
                try:
                    pts_ece = AdaTS(PTS(dim))
                    pts_ece = fitAdaTS(pts_ece, X_val, Y_val,
                                       target_file=os.path.join(res_path, 'PTSece_{}_{}_{:d}.png'.format(model, N, i)),
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
            failed = True
            curr_epochs = 100000
            curr_lr = 1e-3
            while failed:
                try:
                    lts_t = AdaTS(LTS(dim))
                    lts_t = fitAdaTS(lts_t, X_val, Y_val,
                                     target_file=os.path.join(res_path, 'LTS_{}_{}_{:d}.png'.format(model, N, i)),
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
            TSmodels_predictive['LTS'] = lts_t.predictive


            print('\n\tFitting HTS...')
            failed = True
            curr_epochs = 100000
            curr_lr = 1e-3
            while failed:
                try:
                    hts_t = AdaTS(HTS(dim))
                    hts_t = fitAdaTS(hts_t, X_val, Y_val,
                                     target_file=os.path.join(res_path, 'HTS_{}_{}_{:d}.png'.format(model, N, i)),
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
            TSmodels_predictive['HTS'] = hts_t.predictive


            print('\n\tFitting HnLTS...')
            failed = True
            curr_epochs = 100000
            curr_lr = 1e-3
            while failed:
                try:
                    hnlts_t = AdaTS(HnLTS(dim))
                    hnlts_t = fitAdaTS(hnlts_t, X_val, Y_val,
                                       target_file=os.path.join(res_path, 'HnLTS{}_{}_{:d}.png'.format(model, dataset, i)),
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
            TSmodels_predictive['HnLTS'] = hnlts_t.predictive



            acc, ece, bri, nll, mce = compute_metrics(X_test, Y_test, M=50, from_logits=True)
            ece_15 = (compute_ece(softmax(X_test, axis=-1), Y_test))

            eces_15 = []
            eces = []
            bris = []
            nlls = []

            for label, TSmodel in TSmodels_predictive.items():
                TSmetrics = compute_metrics(TSmodel(X_test),
                                            Y_test,
                                            M=50,
                                            from_logits=False)
                eces_15.append(compute_ece(TSmodel(X_test), Y_test))
                eces.append(TSmetrics[1])
                bris.append(TSmetrics[2])
                nlls.append(TSmetrics[3])


            res_ECE15.loc[ix] = [dataset, model, ece_15] + eces_15
            res_ECE.loc[ix] = [N, model, ece] + eces
            res_nll.loc[ix] = [N, model, nll] + nlls
            res_bri.loc[ix] = [N, model, bri] + bris

            ix += 1
            

    res_ECE.to_csv('../results/data_eff/data_eff_ECE_reg_{:d}.csv'.format(i))
    res_nll.to_csv('../results/data_eff/data_eff_NLL_reg_{:d}.csv'.format(i))
    res_bri.to_csv('../results/data_eff/data_eff_BRI_reg_{:d}.csv'.format(i))
    res_ECE15.to_csv('../results/data_eff/data_eff_ECE15_reg_{:d}.csv'.format(i))