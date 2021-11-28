import numpy as np
from scipy import optimize
from mixNmatch_cal import temperature_scaling

import torch

from utils import onehot_encode, binByEntropy, softmax, softplus, sigmoid, entropy


###########################
#### Base Temp-Scaling ####
###########################
def loss_ts(t, *args):
    # unravel args
    logit, label = args

    # temp-scaling
    logit = logit/t.reshape(-1, 1)

    # softmax
    p = np.clip(softmax(logit, axis=1), 1e-20, 1-1e-20)

    # ce-loss
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))
    return ce


def jac_ts(t, *args):
    logits, onehot_target = args

    preds = softmax(logits/t, axis=1)
    grad = -(1/t**2)*np.sum(logits*(preds-onehot_target))
    return grad


class TS:
    def __init__(self, dim):

        self.dim = dim
        
        self.t = 1.0

    def __call__(self, X):

        return X/self.t

    def fit(self, X, y, v=True):

        res = optimize.minimize(loss_ts, self.t ,
                                args=(X, onehot_encode(y, n_classes=self.dim)),
                                jac=jac_ts,
                                options={'disp': v})

        self.t = res.x

    def predictive(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
            
        return softmax(self.__call__(x), axis=-1)


###########################
### Linear Temp-Scaling ###
###########################
def loss_lts(w, *args):
    # unravel args
    logit, label = args
    w, b = w[:-1], w[-1]

    # temp-scaling
    x = (logit @ w + b).reshape([-1, 1])
    t = softplus(x)
    logit = logit/t

    # softmax
    p = np.clip(softmax(logit, axis=1), 1e-20, 1-1e-20)

    # ce-loss
    ce = -np.sum(label*np.log(p))
    return ce


def jac_lts(w, *args):
    # unravel args
    logits, onehot_target = args
    w, b = w[:-1], w[-1]


    x = (logits @ w + b).reshape([-1, 1])
    t = softplus(x)
    preds = softmax(logits/t, axis=1)
    
    grad_w = np.sum(np.sum(-(1/t**2)*logits*(preds-onehot_target), axis=1, keepdims=True) * sigmoid(x)*logits, axis=0)
    grad_b = np.sum(np.sum(-(1/t**2)*logits*(preds-onehot_target), axis=1, keepdims=True) * sigmoid(x))

    return np.append(grad_w, grad_b)


class LTS:
    def __init__(self, dim):

        self.dim = dim
        
        self.w = np.ones((dim, 1))/dim
        self.b = 1.0

    def __call__(self, X):
        t = self.get_T(X)

        return X/t.reshape(-1, 1)

    def fit(self, X, y, v=True):

        res = optimize.minimize(loss_lts, np.append(self.w, self.b) ,
                                args=(X, onehot_encode(y, n_classes=self.dim)),
                                jac=jac_lts,
                                options={'disp': v})

        w = res.x

        self.w, self.b = w[:-1], w[-1]

    def get_T(self, X):
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()

        x = (X @ self.w + self.b).reshape([-1, 1])
        t = softplus(x)
        return t


    def predictive(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
            
        return softmax(self.__call__(x), axis=-1)


##################################
### Entropy-based Temp-Scaling ###
##################################
def loss_hts(w, *args):
    # unravel args
    logit, label, lhs = args
    w, b = w[0], w[1]
    
    # temp-scaling
    x = w*lhs + b
    t = softplus(x)
    logit = logit/t.reshape(-1, 1)

    # softmax
    p = np.clip(softmax(logit, axis=1), 1e-20, 1-1e-20)

    # ce-loss
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))
    return ce


def jac_hts(w, *args):
    logits, onehot_target, lhs = args
    w, b = w[0], w[1]

    x = w*lhs + b
    t = softplus(x)
    preds = softmax(logits/t, axis=1)
    
    grad_w = np.sum(np.sum(-(1/t**2)*logits*(preds-onehot_target), axis=1, keepdims=True) * sigmoid(x)*lhs)
    grad_b = np.sum(np.sum(-(1/t**2)*logits*(preds-onehot_target), axis=1, keepdims=True) * sigmoid(x))
    
    return [grad_w, grad_b]


class HTS:
    def __init__(self, dim):

        self.dim = dim
        
        self.w = .01
        self.b = .1

    def __call__(self, X):
        t = self.get_T(X)

        return X/t.reshape(-1, 1)

    def fit(self, X, y, v=True):

        # precompute entropies
        lhs = np.log(entropy(softmax(X, axis=1), axis=1)/np.log(self.dim))

        res = optimize.minimize(loss_hts, (self.w, self.b) ,
                                args=(X, onehot_encode(y, n_classes=self.dim), lhs),
                                jac=jac_hts,
                                options={'disp': v})

        w = res.x

        self.w, self.b = w[0], w[1]

    def get_T(self, X):
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()

        lhs = np.log(entropy(X, axis=1, from_logits=True)/np.log(self.dim))

        x = self.w*lhs + self.b
        t = softplus(x)
        
        X = X/t.reshape(-1, 1)
        return t


    def predictive(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
            
        return softmax(self.__call__(x), axis=-1)


##################################
### Entropy and Linear Temp-Scaling ###
##################################
def loss_hnlts(w, *args):
    # unravel args
    logit, label, lhs = args
    w, wh, b = w[:-2], w[-2], w[-1]
    
    # temp-scaling
    x = ((logit @ w).reshape([-1, 1]) + wh*lhs + b)
    t = softplus(x)
    logit = logit/t.reshape(-1, 1)

    # softmax
    p = np.clip(softmax(logit, axis=1), 1e-20, 1-1e-20)

    # ce-loss
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))
    return ce


def jac_hnlts(w, *args):
    # unravel args
    logits, onehot_target, lhs = args
    w, wh, b = w[:-2], w[-2], w[-1]
    
    # temp-scaling
    x = ((logits @ w).reshape([-1, 1]) + wh*lhs + b)
    t = softplus(x)
    preds = softmax(logits/t, axis=1)
    
    grad_w = np.sum(np.sum(-(1/t**2)*logits*(preds-onehot_target), axis=1, keepdims=True) * sigmoid(x)*logits, axis=0)
    grad_wh = np.sum(np.sum(-(1/t**2)*logits*(preds-onehot_target), axis=1, keepdims=True) * sigmoid(x)*lhs)
    grad_b = np.sum(np.sum(-(1/t**2)*logits*(preds-onehot_target), axis=1, keepdims=True) * sigmoid(x))
    
    return np.append(grad_w, [grad_wh, grad_b])


class HnLTS:
    def __init__(self, dim):

        self.dim = dim
        
        self.w = np.ones((dim, 1))/dim
        self.wh = .01
        self.b = .1

    def __call__(self, X):
        t = self.get_T(X)

        return X/t.reshape(-1, 1)

    def fit(self, X, y, v=True):

        # precompute entropies
        lhs = np.log(entropy(softmax(X, axis=1), axis=1)/np.log(self.dim))

        res = optimize.minimize(loss_hnlts, np.append(self.w, [self.wh, self.b]) ,
                                args=(X, onehot_encode(y, n_classes=self.dim), lhs),
                                jac=jac_hnlts,
                                options={'disp': v})

        w = res.x

        self.w, self.wh, self.b = w[:-2], w[-2], w[-1]

    def get_T(self, X):
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()

        lhs = np.log(entropy(X, axis=1, from_logits=True)/np.log(self.dim))

        x = ((X @ self.w).reshape([-1, 1]) + self.wh*lhs + self.b)
        t = softplus(x)
        
        X = X/t.reshape(-1, 1)
        return t


    def predictive(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
            
        return softmax(self.__call__(x), axis=-1)


#######################################
### Hist-Entropy-based Temp-Scaling ###
#######################################
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

        N, dim = X.shape

        self.Ts = np.ones(self.M)
        for i, ix in enumerate(ixs):
            if any(ix):
                try:
                    self.Ts[i] = temperature_scaling(X[ix], onehot_encode(y[ix], n_classes=dim), 'ce')
                except Exception as e:
                    continue

    def __call__(self, x):
        ## Compute entropy of samples
        Hs = entropy(x, axis=1, from_logits=True).flatten()

        N, dim = x.shape

        if self.normalize:
            Hs /= np.log(dim)
        
        X_cal = np.zeros(x.shape) + x
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=Hs) & (Hs<high)
            if any(ix):
                X_cal[ix] = x[ix]/self.Ts[i]
                
        return X_cal

    def get_T(self, x):
        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()

        H_val = entropy(softmax(x, axis=1), axis=1)
        
        Ts = torch.zeros(x.shape[0])
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=H_val) & (H_val<high)
            if any(ix):
                Ts[ix] = self.Ts[i]

        return Ts.detach().numpy()

    def predictive(self, x):
        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()
            
        return softmax(self.__call__(x), axis=1)


#############################
### Bin-wise Temp-Scaling ###
#############################

class BTS:
    """Implements Bin-wise Temperature Scaling with equal samples per bin from https://arxiv.org/abs/1908.11528"""
    def __init__(self, M=50):
        self.M = M

    def fit(self, X, y, v=False):

        if torch.is_tensor(X):
            X = X.cpu().detach().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        ixs, lims = self.get_lims(X)
        self.lims = lims

        self.Ts = np.ones(self.M)
        N, dim = X.shape       

        for i, ix in enumerate(ixs):
            if any(ix):
                try:
                    self.Ts[i] = temperature_scaling(X[ix], onehot_encode(y[ix], n_classes=dim), 'ce')
                    print('Fitted bin {:d}, with T: {:.2f}'.format(i, self.Ts[i]), end="\r")
                except Exception as e:
                    print(e)
                    continue

    def get_lims(self, logits):
        confs = np.max(softmax(logits, axis=1), axis=1)

        low_confs = confs[confs<0.999]

        lims = np.quantile(low_confs, np.linspace(0, 1, self.M))

        lims = np.hstack((lims, [1]))

        ## Compute idxs entropy per bin
        ixs = []
        for i, (low, high) in enumerate(zip(lims[:-1], lims[1:])):
            ix = (low<confs) & (confs<=high)
            
            ixs.append(ix)

        return ixs, lims

    def __call__(self, x):
        ## Compute entropy of samples
        confs = np.max(softmax(x, axis=1), axis=1)

        N, dim = x.shape
        
        X_cal = np.zeros(x.shape) + x
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=confs) & (confs<high)
            if any(ix):
                X_cal[ix] = x[ix]/self.Ts[i]
                
        return X_cal

    def get_T(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        confs = np.max(softmax(x, axis=1), axis=1)
        
        Ts = torch.zeros(x.shape[0])
        
        for i, (low, high) in enumerate(zip(self.lims[:-1], self.lims[1:])):
            ix = (low<=confs) & (confs<high)
            if any(ix):
                Ts[ix] = self.Ts[i]

        return Ts.detach().numpy()

    def predictive(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
            
        return softmax(self.__call__(x), axis=1)