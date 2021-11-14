import time
from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import cross_entropy

from models import NanValues, AdaTS, HistTS
from utils import torch_entropy, torch_softmax



def fitAdaTS(adaTS, X, Y,
             epochs=1000,
             batch_size=None,
             lr=1e-2,
             optimizer='adam',
             weight_decay=0,
             v=False,
             target_file=None,
             dev='cpu'):

    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32)

    if not torch.is_tensor(Y):
        Y = torch.as_tensor(Y, dtype=torch.long)

    N = X.shape[0]

    # Pre-compute optimum T
    if adaTS.prescale:
        if v:
            print('Finding optimum Temperature')
        optim = torch.optim.SGD([adaTS.T], lr=1e-2)

        _T = np.zeros(10)
        e=0
        t0 = time.time()
        while True:
            loss = cross_entropy(adaTS(X, pretrain=True), Y, reduction='mean')

            if loss != loss:
                raise NanValues("Aborting training due to nan values")

            optim.zero_grad()
            loss.backward()
            optim.step()

            if v and e % 10 == 4:
                print('On epoch: {:d}, NLL: {:.3e}, '.format(e, N*loss.item())
                        + 'Temp: {:.3f}, '.format(adaTS.T.item())
                        + 'at time: {:.2f}s'.format(time.time() - t0), end="\r")
            _T[e%10] = adaTS.T.item()
            e += 1

            if e>10 and np.mean(np.abs(np.diff(_T)))<1e-7:
                if v:
                    print('\nFound optimum Temperature: {:.3f}'.format(adaTS.T.item()))
                break

    optim = torch.optim.Adam(adaTS.modelT.parameters(), lr=lr, weight_decay=weight_decay) \
        if optimizer=='adam' else \
            torch.optim.SGD(adaTS.modelT.parameters(), lr=lr, weight_decay=weight_decay)

    if batch_size is None:
        batch_size=N
    n_steps = int(np.ceil(N/batch_size))
    e=0
    t0 = time.time()

    if target_file is not None:
        running_nll = []

    adaTS.to(dev)

    while e<epochs:
        # Shuffle data before each epoch
        perm = np.random.permutation(N)
        X = X[perm]
        Y = Y[perm]

        nll = 0
        for s in range(n_steps):
            x = X[s*batch_size:min((s+1)*batch_size, N)]
            y = Y[s*batch_size:min((s+1)*batch_size, N)]

            if dev != 'cpu':
                x = x.to(dev)
                y = y.to(dev)

            n = x.shape[0]

            logits = adaTS(x)

            _nll = cross_entropy(logits, y, reduction='mean')

            if _nll != _nll:
                raise NanValues("Aborting training due to nan values")

            # Train step
            optim.zero_grad()
            _nll.backward()
            optim.step()

            nll += n*_nll.item()

        if v and ((e % 10) == 4):
            print('On epoch: {:d}, NLL: {:.3e}, '.format(e, nll)
                    + 'at time: {:.2f}s'.format(time.time() - t0), end="\r")
        e += 1

        if target_file is not None:
            running_nll.append(nll)

    if target_file is not None:
        fig, ax = plt.subplots()
        ax.plot(running_nll)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('NLL')
        try:
            fig.savefig(target_file, dpi=300)

        except:
            print('Couldnt save training to {}'.format(target_file))

    return adaTS.cpu()




def fitCV_AdaTS(modelT, X, Y,
                lrs=[1e-2, 1e-1],
                weight_decays=[1e-3, 1e-2, 1e-1],
                val_split=0.2,
                iters=5,
                batch_size=1000,
                epochs=1000,
                v=True,
                target_file=None,
                criterion=torch.nn.functional.cross_entropy):
    if not isinstance(lrs, Iterable):
        lrs = [lrs]

    if not isinstance(weight_decays, Iterable):
        weight_decays = [weight_decays]

    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32)

    if not torch.is_tensor(Y):
        Y = torch.as_tensor(Y, dtype=torch.long)

        N, dim = X.shape

    if len(lrs)==1 and len(weight_decays)==1:
        model = AdaTS(modelT(dim))
        model = fitAdaTS(model, X, Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lrs[0],
                        optimizer='adam',
                        weight_decay=weight_decays[0],
                        v=v,
                        target_file=target_file)

        return model



    best = np.inf
    best_lr = lrs[0]
    best_wd = weight_decays[0]

    for lr in lrs:
        for wd in weight_decays:
            if v:
                print('Validating configuration lr:{:.2e}, weight decay:{:.2e}'.format(lr, wd))
            val_loss = 0
            for _ in range(iters):
                ix_random = np.random.permutation(N)
                ix_val, ix_train = ix_random[:int(val_split*N)], ix_random[int(val_split*N):]

                model = AdaTS(modelT(dim))

                model = fitAdaTS(model, X[ix_train], Y[ix_train], epochs=epochs, batch_size=batch_size, lr=lr, optimizer='adam', weight_decay=wd, v=False)
                with torch.no_grad():
                    Z = model(X[ix_val])
                    val_loss += criterion(Z, Y[ix_val])

            if val_loss < best:
                if v:
                    print('Found new best configuration with mean validation loss: {:.3f}'.format(val_loss/iters))
                best = val_loss
                best_lr = lr
                best_wd = wd
    
    model = AdaTS(modelT(dim))
    model = fitAdaTS(model, X, Y,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=best_lr,
                     optimizer='adam',
                     weight_decay=best_wd,
                     v=v,
                     target_file=target_file)

    return model


def fitHistTS(X, Y,
              val_split=0.2,
              iters=5,
              Ms=[15, 50, 100],
              v=True,
              criterion=torch.nn.functional.cross_entropy):

    if not isinstance(Ms, Iterable):
        Ms = [Ms]

    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32)

    if not torch.is_tensor(Y):
        Y = torch.as_tensor(Y, dtype=torch.long)

    N, dim = X.shape

    best = np.inf

    for M in Ms:
        if v:
            print('Validating configuration M:{:d}'.format(M))
        val_loss = 0
        for _ in range(iters):
            ix_random = np.random.permutation(N)
            ix_val, ix_train = ix_random[:int(val_split*N)], ix_random[int(val_split*N):]

            model = HistTS(M=M)
            model.fit(X[ix_train], Y[ix_train])
            
            with torch.no_grad():
                Z = model(X[ix_val])
                val_loss += criterion(Z, Y[ix_val])

        if val_loss < best:
            if v:
                print('Found new best configuration with mean validation loss: {:.3f}'.format(val_loss/iters))
            best = val_loss
            bestM = M

    hisTS = HistTS(M=bestM)
    hisTS.fit(X, Y)

    return hisTS