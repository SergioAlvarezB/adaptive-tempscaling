import time

import numpy as np
import torch
from torch import nn

from torch.nn.functional import softplus

from adaptive_tempscaling.utils import NanValues, compute_metrics

class BayesianTempScaling_N(nn.Module):
    """
    VI approximation to temp-scaling with a normal prior and posterior 
    over the tranformed latent space.
    """
    def __init__(self, pmean=torch.Tensor([0.0]), plog_var=torch.Tensor([0.0]), transform='exp'):
        super(BayesianTempScaling_N, self).__init__()

        if not torch.is_tensor(pmean):
            pmean = torch.as_tensor(pmean, dtype=torch.float32)

        if not torch.is_tensor(plog_var):
            plog_var = torch.as_tensor(plog_var, dtype=torch.float32)


        # Init Params
        self.iT_mean = nn.Parameter(torch.Tensor([1.0]))
        self.iT_log_var = nn.Parameter(torch.Tensor([0.0]))
        self.pmean = nn.Parameter(pmean, requires_grad=False)
        self.plog_var = nn.Parameter(plog_var, requires_grad=False)

        # Tranformation
        if transform == 'exp':
            self.iTransform = torch.exp
        elif transform == 'softplus':
            self.iTransform = softplus
        elif callable(transform):
            self.iTransform = transform
        else:
            raise ValueError('transform argument {} not recognized'.format(transform))

        # Loss function
        self.CE = torch.nn.functional.cross_entropy
        self.softmax = nn.functional.softmax

    def evaluate(self, X, y, n_samples=1000, step=10):

        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)

        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.int64)


        is_list = True
        if not isinstance(n_samples, (tuple, list)):
            is_list = False
            n_samples = [n_samples]
        n_samples.sort()
        cum = 0
        preds = 0.
        results = []
        for k in n_samples:
            curr = int(k-cum)
            preds = (curr*self.predictive(X, n_samples=curr, step=step)
                     + preds*cum)/(cum + curr)
            cum += curr
            ACC, ECE, BRI, NLL, MCE = compute_metrics(preds.cpu(),
                                                              y.cpu())
            results.append((ACC, ECE, MCE, BRI.item(), NLL.item()))

        if is_list:
            return results
        else:
            return results[0]

    def ELBO(self, x, y, w_nll=1., beta=1., MC_samples=10, step=100):
        rem = MC_samples

        NLL = 0.0

        while rem:
            k = min(step, rem)
            rem -= k
            x_r = torch.cat(k*[x])
            NLL +=  self.CE(self.forward(x_r), y.repeat(k), reduction='sum')

        NLL /= MC_samples
        NLL *= w_nll
        D = self.divergence()
        ELBO = NLL + beta*D

        return ELBO, NLL, D

    def fit(self,
            X,
            Y,
            epochs,
            batch_size,
            MC_samples,
            beta=1.,
            optimizer=None,
            lr=1e-2,
            warm_up=0,
            v=False):

        N = X.shape[0]

        n_steps = int(np.ceil(N/batch_size))

        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        t0 = time.time()
        nll, loss = [], []
        d = []
        acc, ece, mce = [], [], []
        acc20, ece20, mce20 = [], [], []

        e = 0

        while True:

            _nll = 0
            _loss_cum = 0
            _d = 0

            # Shuffle data before each epoch
            perm = np.random.permutation(N)
            X = X[perm]
            Y = Y[perm]

            for s in range(n_steps):
                x = X[s*batch_size:min((s+1)*batch_size, N)]
                y = Y[s*batch_size:min((s+1)*batch_size, N)]

                n = x.shape[0]

                self.train()
                _loss, _NLL, _D = self.ELBO(x, y,
                                            w_nll=(N/n),
                                            beta=beta,
                                            MC_samples=MC_samples)

                _loss = _loss if e > warm_up else _NLL

                if _loss != _loss:
                    raise NanValues("Aborting training due to nan values")

                _loss_cum += _loss.item()
                _nll += _NLL.item()
                _d += _D.item()

                # Train step
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

            _loss_cum /= n_steps
            _nll /= n_steps
            _d /= n_steps

            self.eval()
            if e % 20 == 0:
                _acc, _ece, _mce, _, _ = self.evaluate(X, Y, n_samples=100, step=1)
                acc20.append(_acc)
                ece20.append(_ece)
                mce20.append(_mce)
            else:
                _acc, _ece, _mce, _, _ = self.evaluate(X, Y, n_samples=1)
                acc.append(_acc)
                ece.append(_ece)
                mce.append(_mce)

            loss.append(_loss_cum)
            nll.append(_nll)
            d.append(_d)

            if v and e % 5 == 4:
                print('On epoch: {:d}, NLL: {:.3e}, '.format(e, nll[-1])
                      + 'D: {:.3e}, loss(diff): {:.3f}'.format(d[-1], np.max(np.abs(np.diff(loss[-5:]))))
                      + ', at time: {:.2f}s'.format(time.time() - t0), end="\r")
            e += 1

            # Stop condition
            if (epochs is not None and e >= epochs) \
                or (e>5 and np.max(np.abs(np.diff(loss[-5:]))) < 1e-6):
                break


        h = {
            'nll': np.array(nll),
            'divergence': np.array(d),
            'acc': np.array(acc),
            'ece': np.array(ece),
            'mce': np.array(mce),
            'acc20': np.array(acc20),
            'ece20': np.array(ece20),
            'mce20': np.array(mce20),
        }

        return h

    def forward(self, x):

        log_iT = self.iT_mean \
            + torch.randn_like(self.iT_mean)*torch.sqrt(torch.exp(self.iT_log_var))

        iT = self.iTransform(log_iT)

        return x*iT
    
    def sample_posterior(self, N=1000):
        log_iT = self.iT_mean \
            + torch.randn((N, *self.iT_mean.size()))*torch.sqrt(torch.exp(self.iT_log_var))

        return self.iTransform(log_iT)


    def MAP(self, x):

        log_iT = self.iT_mean

        iT = self.iTransform(log_iT)

        return x*iT

    def prior_forward(self, x):

        log_iT = self.pmean \
            + torch.randn_like(self.pmean)*torch.sqrt(torch.exp(self.plog_var))

        iT = self.iTransform(log_iT)

        return x*iT

    def predictive(self,
                   x,
                   n_samples=100,
                   step=10,
                   reduce=True,
                   prior=False,
                   MAP=False):

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        self.eval()

        if MAP:
            z = self.MAP(x)
            preds = self.softmax(z, dim=1)
            return preds


        preds = 0.0 if reduce else []
        rem = n_samples
        k = 0
        while rem:
            k = min(step, rem)
            rem -= k
            x_r = torch.cat(k*[x])
            if prior:
                preds_r = self.softmax(self.prior_forward(x_r).detach(), dim=1)
            else:
                preds_r = self.softmax(self.forward(x_r).detach(), dim=1)
            if reduce:
                preds += sum(torch.split(preds_r, x.shape[0], dim=0))
            else:
                preds += torch.split(preds_r, x.shape[0], dim=0)
        return (preds/n_samples).float() if reduce \
            else torch.stack(preds).permute(1,0,2)


    def divergence(self):
        pvar = torch.exp(self.plog_var)
        qvar = torch.exp(self.iT_log_var)

        KL = 0.5*((qvar/pvar)*((self.iT_mean - self.pmean)**2 / pvar - 1) + self.plog_var - self.iT_log_var)

        return KL