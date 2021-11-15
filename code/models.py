import time

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax, cross_entropy, softplus

from utils import torch_entropy, binByEntropy


class NanValues(Exception):
    """"Custom exception to signal failed training"""


## Non-Parametric Temp-Scaling


class BTS:
    """Implements Bin-wise Temperature Scaling with equal samples per bin from https://arxiv.org/abs/1908.11528"""
    def __init__(self, M=50):
        self.M = M

    def fit(self, X, y, v=False):

        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)

        with torch.no_grad():
            ixs, lims = self.get_lims(X)
        self.lims = lims

        self.Ts = np.ones(self.M)
        for i, ix in enumerate(ixs):
            if any(ix):
                try:
                    ts_aux = TempScaling()
                    ts_aux.fit(X[ix], y[ix], v=v);
                    self.Ts[i] = ts_aux.T.detach().numpy()
                except Exception as e:
                    continue


    def get_lims(self, logits):
        confs, preds = torch.max(softmax(logits, dim=1), dim=1)

        low_confs = confs[confs<0.999]

        lims = np.quantile(low_confs.detach().cpu().numpy(), np.linspace(0, 1, self.M))

        lims = np.hstack((lims, [1]))

        ## Compute idxs entropy per bin
        ixs = []
        for i, (low, high) in enumerate(zip(lims[:-1], lims[1:])):
            ix = (low<confs) & (confs<=high)
            
            ixs.append(ix.numpy())

        return ixs, lims


    def __call__(self, x):
        ## Compute entropy of samples
        with torch.no_grad():
            confs, preds = torch.max(softmax(x, dim=1), dim=1)

        N, dim = x.shape
        
        X_cal = torch.zeros_like(x) + x
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=confs) & (confs<high)
            if any(ix):
                X_cal[ix] = x[ix]/self.Ts[i]
                
        return X_cal

    def get_T(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        with torch.no_grad():
            confs = torch.max(softmax(x, dim=1), dim=1)
        
        Ts = torch.zeros(x.shape[0])
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=confs) & (confs<high)
            if any(ix):
                Ts[ix] = self.Ts[i]

        return Ts.detach().numpy()

    def predictive(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        return softmax(self.__call__(x), dim=-1)




class HistTS:

    def __init__(self, M=15, mode='same', normalize=True):
        self.M = M
        assert mode in ['log', 'same']
        self.mode = mode
        self.Ts = None
        self.normalize = normalize

    def fit(self, X, y, M=None, mode=None, v=False):

        if M is not None:
            self.M = M
        if mode is not None:
            assert mode in ['log', 'same']
            self.mode = mode

        ixs, lims = binByEntropy(X, M=self.M, mode=self.mode, normalize=self.normalize)
        self.lims = lims

        self.Ts = np.ones(self.M)
        for i, ix in enumerate(ixs):
            if any(ix):
                try:
                    ts_aux = TempScaling()
                    ts_aux.fit(X[ix], y[ix], v=v);
                    self.Ts[i] = ts_aux.T.detach().numpy()
                except Exception as e:
                    continue

    def __call__(self, x):
        ## Compute entropy of samples
        Hs = torch_entropy(x)

        N, dim = x.shape

        if self.normalize:
            Hs /= np.log(dim)
        
        X_cal = torch.zeros_like(x) + x
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=Hs) & (Hs<high)
            if any(ix):
                X_cal[ix] = x[ix]/self.Ts[i]
                
        return X_cal

    def get_T(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        H_val = torch_entropy(x)
        
        Ts = torch.zeros(x.shape[0])
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=H_val) & (H_val<high)
            if any(ix):
                Ts[ix] = self.Ts[i]

        return Ts.detach().numpy()

    def predictive(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        return softmax(self.__call__(x), dim=-1)



class TempScaling(nn.Module):

    def __init__(self):
        super(TempScaling, self).__init__()

        # Init temperature
        self.T = nn.Parameter(torch.Tensor([1.0]))

    def fit(self, X, y, lr=1e-1, v=False, cte_epochs=5):

        N = X.shape[0]

        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)

        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long)

        optim = torch.optim.SGD(self.parameters(), lr=lr)

        self.train()

        _T = np.zeros(cte_epochs)
        e=0
        t0 = time.time()
        while True:
            loss = cross_entropy(self.forward(X), y, reduction='mean')

            if loss != loss:
                raise NanValues("Aborting training due to nan values")

            optim.zero_grad()
            loss.backward()
            optim.step()

            if v and e % 10 == 4:
                print('On epoch: {:d}, NLL: {:.3e}, '.format(e, N*loss.item())
                     + 'Temp: {:.3f}, '.format(self.T.item())
                      + 'at time: {:.2f}s'.format(time.time() - t0), end="\r")
            _T[e%cte_epochs] = self.T.item()
            e += 1

            if e>cte_epochs and np.mean(np.abs(np.diff(_T)))<1e-7:
                break

        return self.forward(X)

    def forward(self, x):
        return x/torch.abs(self.T)

    def predictive(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        return softmax(self.forward(x), dim=-1)


class EnsembleTS(nn.Module):
    def __init__(self, dim):
        super(EnsembleTS, self).__init__()
        # Init params

        self.T = nn.Parameter(torch.Tensor([1.0]))
        self.W = nn.Parameter(torch.rand(3))
        self.dim = dim

    def fit(self, X, y, lr=1e-2, v=False, cte_epochs=5):

        N = X.shape[0]

        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)

        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long)

        optim = torch.optim.SGD(self.parameters(), lr=lr)

        self.train()

        _T = np.zeros(cte_epochs)
        e=0
        t0 = time.time()
        print('Finding optimum T')
        while True:
            loss = cross_entropy(self.forward(X), y, reduction='mean')

            if loss != loss:
                raise NanValues("Aborting training due to nan values")

            optim.zero_grad()
            loss.backward()
            optim.step()

            if v and e % 10 == 4:
                print('On epoch: {:d}, NLL: {:.3e}, '.format(e, N*loss.item())
                     + 'Temp: {:.3f}, '.format(self.T.item())
                      + 'at time: {:.2f}s'.format(time.time() - t0), end="\r")
            _T[e%cte_epochs] = self.T.item()
            e += 1

            if e>cte_epochs and np.mean(np.abs(np.diff(_T)))<1e-7:
                break

        return self.forward(X)

    def forward(self, x):
        # Obtain Weights
        W_u = softplus(self.W)
        W = W_u/torch.sum(W_u)

        x1 = softmax(x/self.T, dim=1)
        x2 = softmax(x, dim=1)

        c = W[0]*x1 + W[1]*x2 + W[2]*(1/self.dim)

        return torch.log(c)

    def predictive(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        return softmax(self.forward(x), dim=-1)

# #### Generic Temp-Scaling model

class AdaTS(nn.Module):
    
    def __init__(self, modelT):
        super(AdaTS, self).__init__()

        self.prescale = modelT.prescale
        if self.prescale:
            # Init temperature
            self.T = nn.Parameter(torch.Tensor([1.0]))

        self.modelT = modelT

    def forward(self, x, pretrain=False):
        if pretrain:
            return x/self.T
            
        Ts = self.modelT(x)
        if self.prescale:
            Ts *= self.T

        return x/Ts.view(-1, 1)

    def get_T(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        Ts = self.modelT(x)

        if self.prescale:
            Ts *= self.T

        return Ts.detach().numpy()

    def predictive(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        return softmax(self.forward(x), dim=-1)


# #### Temperature Models

class ScaleT(nn.Module):
    
    def __init__(self, dim):
        super(ScaleT, self).__init__()
        
        self.prescale=True

        self.b = nn.Parameter(torch.Tensor([1.0]))
        self.W = nn.Parameter(torch.randn(dim)/(100*dim))
        self.dim = dim
        
    def forward(self, x):
        T = (torch.tanh((x @ self.W)/self.dim + self.b) + 1)
        return T


class LinearT(nn.Module):
    def __init__(self, dim, norm=True):
        super(LinearT, self).__init__()
        self.prescale=False
        # Init params
        self.b = nn.Parameter(torch.Tensor([1.0]))
        self.W = nn.Parameter(torch.randn(dim)/(dim))

        self.dim = dim
        self.norm = norm
        
    def forward(self, x):
        if self.norm:
            W = self.W/torch.norm(self.W)
        else:
            W = self.W
        T = softplus((x @ W) + self.b)
        return T


class HbasedT(nn.Module):
    def __init__(self, dim):
        super(HbasedT, self).__init__()
        self.prescale=False
        # Init params
        self.b = nn.Parameter(torch.Tensor([1.0]))
        self.w = nn.Parameter(torch.Tensor([1.0]))
        self.t = nn.Parameter(torch.Tensor([1.0]))

        self.dim = dim
    
    def forward(self, x):
        with torch.no_grad():
            H = torch_entropy(x)/np.log(self.dim)

        T = (torch.tanh(H * self.w + self.b) + 1) * self.t
        return T


class HlogbasedT(nn.Module):
    def __init__(self, dim):
        super(HlogbasedT, self).__init__()
        self.prescale=False
        # Init params
        self.b = nn.Parameter(torch.Tensor([.1]))
        self.w = nn.Parameter(torch.Tensor([.1]))

        self.dim = dim
    
    def forward(self, x):
        with torch.no_grad():
            logH = torch.log(torch_entropy(x)/np.log(self.dim))

        T = softplus(logH * self.w + self.b)
        return T


class HnLinearT(nn.Module):
    def __init__(self, dim):
        super(HnLinearT, self).__init__()
        self.prescale=False

        # Init params
        self.b = nn.Parameter(torch.Tensor([.1]))
        self.wh = nn.Parameter(torch.Tensor([.1]))
        self.W = nn.Parameter(torch.randn(dim)/(dim))

        self.dim = dim

    def forward(self, x):
        with torch.no_grad():
            logH = torch.log(torch_entropy(x)/np.log(self.dim))

        T = softplus(x @ self.W + logH * self.wh + self.b)
        return T


class DNNbasedT(nn.Module):
    def __init__(self, dim, hs=None):
        super(DNNbasedT, self).__init__()
        self.prescale=False
        # Init params

        if hs is None:
            hs = [int(np.sqrt(dim))]

        hs = [dim] + hs + [1]

        self.fcs = nn.ModuleList([nn.Linear(inp, out) for (inp, out) in zip(hs[:-1], hs[1:])])

        self.dim = dim
        
    def forward(self, x):

        for fc in self.fcs[:-1]:
            x = torch.relu(fc(x))
            
        T = softplus(self.fcs[-1](x))
        return T


# ### NN models

class LeNet5(nn.Module):

    def __init__(self, dim, input_channels=1):
        super(LeNet5, self).__init__()
        
        self.cnn = nn.Sequential(            
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=3),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, dim),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return z

    def predictive(self, x):
        return softmax(self.forward(x), dim=1)
