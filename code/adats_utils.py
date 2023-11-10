import time
from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import cross_entropy

from models import NanValues, AdaTS, HistTS


def ece_loss(logits, target, M=10, reduction='sum'):
    """"Computes ECE based loss as used in https://arxiv.org/abs/2102.12182"""

    probs = torch.softmax(logits, dim=1)
    N = probs.shape[0]

    confs, preds = torch.max(probs, dim=1)


    # Generate intervals
    limits = np.linspace(0, 1, num=M+1)
    lows, highs = limits[:-1], limits[1:]

    ece = 0

    for low, high in zip(lows, highs):

        ix = (low < confs) & (confs <= high)
        n = torch.sum(ix)
        if n<1:
            continue

        curr_preds = preds[ix]
        curr_confs = confs[ix]
        curr_target = target[ix]

        curr_acc = np.mean(curr_preds.cpu().numpy() == curr_target.cpu().numpy())

        ece += n*torch.sqrt((torch.mean(curr_confs)-curr_acc)**2)

    if reduction == 'mean':
        ece /= N

    return ece



def fitAdaTS(adaTS, X, Y,
             epochs=1000,
             batch_size=None,
             lr=1e-2,
             optimizer='adam',
             loss='nll',
             weight_decay=0,
             v=False,
             target_file=None,
             dev='cpu'):


    assert loss in ['ece', 'nll']

    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float64)

    if not torch.is_tensor(Y):
        Y = torch.as_tensor(Y, dtype=torch.long)

    N = X.shape[0]

    if loss == 'nll':
        loss_f = cross_entropy
    elif loss == 'ece':
        loss_f = ece_loss

    # Pre-compute optimum T
    if adaTS.prescale:
        if v:
            print('Finding optimum Temperature')
        optim = torch.optim.SGD([adaTS.T], lr=1e-2)

        _T = np.zeros(10)
        e=0
        t0 = time.time()
        while True:
            _loss = loss_f(adaTS(X, pretrain=True), Y, reduction='mean')

            if _loss != _loss:
                raise NanValues("Aborting training due to nan values")

            optim.zero_grad()
            _loss.backward()
            optim.step()

            if v and e % 10 == 4:
                print('On epoch: {:d}, NLL: {:.3e}, '.format(e, N*_loss.item())
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
            torch.optim.SGD(adaTS.modelT.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    scheduler = ReduceLROnPlateau(optim, 'min', patience=100, cooldown=50)

    if batch_size is None or batch_size>N:
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

        cum_loss = 0
        for s in range(n_steps):
            x = X[s*batch_size:min((s+1)*batch_size, N)]
            y = Y[s*batch_size:min((s+1)*batch_size, N)]

            if dev != 'cpu':
                x = x.to(dev)
                y = y.to(dev)

            n = x.shape[0]

            logits = adaTS(x)

            _loss = loss_f(logits, y, reduction='mean')

            if _loss != _loss:
                raise NanValues("Aborting training due to nan values")

            # Train step
            optim.zero_grad()
            _loss.backward()
            optim.step()

            cum_loss += n*_loss.item()

        if loss == 'ece':
            cum_loss /= N

        if v and ((e % 10) == 4):
            print('On epoch: {:d}, loss: {:.3e}, '.format(e, cum_loss)
                    + 'at time: {:.2f}s'.format(time.time() - t0), end="\r")
        e += 1

        
        scheduler.step(cum_loss)
        if optim.param_groups[0]["lr"] < 1e-7:
            print("\nFinish training, convergence reached. Loss: {:.2f} \n".format(cum_loss))
            break

        if target_file is not None:
            running_nll.append(cum_loss)

    if target_file is not None:
        fig, ax = plt.subplots()
        ax.plot(running_nll)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        try:
            fig.savefig(target_file, dpi=300)

        except:
            print('Couldnt save training to {}'.format(target_file))

        plt.close()

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