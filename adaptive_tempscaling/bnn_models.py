import time
import numpy as np
import torch
from torch import nn

from adaptive_tempscaling.utils import NanValues, compute_metrics


def gauss_KLD(qmu, qlog_var, pmu, plog_var):
    return 0.5 * torch.sum(torch.exp(qlog_var - plog_var)
                           + (qmu - pmu)**2/torch.exp(plog_var) - 1
                           + (plog_var - qlog_var))


class Linear_LR(nn.Module):

    def __init__(self, input_dim, output_dim, pmean, plog_var):
        super(Linear_LR, self).__init__()

        # Prior distribution
        self.plog_var = nn.Parameter(torch.Tensor([plog_var]),
                                     requires_grad=False)
        self.b_pmean = nn.Parameter(torch.zeros(output_dim,) + pmean,
                                    requires_grad=False)
        self.w_pmean = nn.Parameter(torch.zeros(input_dim, output_dim) + pmean,
                                    requires_grad=False)

        # Initialize parameters

        # The log_var is initialize to -10 following
        # github.com/JeremiasKnoblauch/GVIPublic/blob/master/BayesianNN/GVI_BNN.py
        self.w_mean = nn.Parameter(torch.randn(input_dim, output_dim))
        self.w_log_var = nn.Parameter(torch.zeros(input_dim, output_dim) - 10.)

        self.b_mean = nn.Parameter(torch.randn(output_dim,))
        self.b_log_var = nn.Parameter(torch.zeros(output_dim,) - 10.)

    def forward(self, x, LR=True):

        if LR:
            # Local Reparametrization
            Z_mu = (torch.mm(x, self.w_mean)) + self.b_mean

            Z_sigma = torch.sqrt(torch.mm(x**2, torch.exp(self.w_log_var))
                                + torch.exp(self.b_log_var))

            Z = Z_mu + torch.randn_like(Z_mu)*Z_sigma

        else:
            # Per Mini-batch weight sampling
            w = self.w_mean + \
                torch.randn_like(self.w_mean)*torch.sqrt(torch.exp(self.w_log_var))

            b = self.b_mean + \
                torch.randn_like(self.b_mean)*torch.sqrt(torch.exp(self.b_log_var))

            Z = (torch.mm(x, w)) + b

        return Z

    def prior_forward(self, x):
        # Per Mini-batch weight sampling
        w = self.w_pmean + \
            torch.randn_like(self.w_pmean)*torch.sqrt(torch.exp(self.plog_var))

        b = self.b_pmean + \
            torch.randn_like(self.b_pmean)*torch.sqrt(torch.exp(self.plog_var))

        Z = (torch.mm(x, w)) + b

        return Z

    def MAP(self, x):
        Z_mu = (torch.mm(x, self.w_mean)) + self.b_mean

        return Z_mu

    def get_total_params(self):

        # Return total number of parameters
        return self.w_mean.numel()*2 + self.b_mean.numel()*2

    def get_KLcollapsed_posterior(self):
        # Check If the parameters has collapsed to the prior
        w_kl = 0.5 * (torch.exp(self.w_log_var - self.plog_var)
                      + (self.w_mean - self.w_pmean)**2/torch.exp(self.plog_var)
                      - 1 + (self.plog_var - self.w_log_var))

        w = (w_kl <= 7.5e-05).sum()
        b_kl = 0.5 * (torch.exp(self.b_log_var - self.plog_var)
                      + (self.b_mean - self.b_pmean)**2/torch.exp(self.plog_var)
                      - 1 + (self.plog_var - self.b_log_var))

        b = (b_kl <= 7.5e-05).sum()

        return (w+b).detach().cpu().numpy()

    def get_collapsed_posterior(self):
        # Check If the parameters has collapsed to the prior
        w = ((self.w_mean <= self.w_pmean + 0.01)
             & (self.w_mean >= self.w_pmean - 0.01)
             & (self.w_log_var <= self.plog_var + 0.01)
             & (self.w_log_var >= self.plog_var - 0.01)).float().sum()

        b = ((self.b_mean <= self.b_pmean + 0.01)
             & (self.b_mean >= self.b_pmean - 0.01)
             & (self.b_log_var <= self.plog_var + 0.01)
             & (self.b_log_var >= self.plog_var - 0.01)).float().sum()

        return (w+b).detach().cpu().numpy()

    def update_prior(self):
        # Prior distribution
        with torch.no_grad():
            self.w_pmean.copy_(self.w_mean.data)
            self.b_pmean.copy_(self.b_mean.data)


            # self.w_mean.copy_(torch.randn_like(self.w_mean.data))
            # self.b_mean.copy_(torch.randn_like(self.b_mean.data))

    def get_flatten_variances(self):

        variances = np.concatenate((self.w_log_var.data.detach().cpu().numpy().flatten(),
                                    self.b_log_var.data.detach().cpu().numpy().flatten()))

        return variances

class BNN_GVILR(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 pmean=torch.Tensor([0.0]),
                 plog_var=torch.Tensor([0.0]),
                 hidden_size=[],
                 low_rank=False):
        super(BNN_GVILR, self).__init__()

        # Get general divergence
        self.divergence = gauss_KLD

        self.lowr = low_rank

        d_out = output_dim-1 if low_rank else output_dim

        # DNN architecture
        hs = [input_dim] + hidden_size + [d_out]

        # Loss function
        self.CE = torch.nn.functional.cross_entropy
        self.softmax = nn.functional.softmax

        # Initialize layers
        self.layers = nn.ModuleList([Linear_LR(inp, out, pmean, plog_var)
                                     for inp, out in zip(hs[:-1], hs[1:])])


    def forward(self, x, LR=True):
        z = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                z = torch.relu(z)
            z = layer(z, LR=LR)

        if self.lowr:
            zp = torch.zeros((z.shape[0], z.shape[1]+1), device=z.device)
            zp[:, :z.shape[1]] = z
            z = zp

        return z

    def prior_forward(self, x):
        z = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                z = torch.relu(z)
            z = layer.prior_forward(z)

        if self.lowr:
            zp = torch.zeros((z.shape[0], z.shape[1]+1), device=z.device)
            zp[:, :z.shape[1]] = z
            z = zp

        return z

    def compute_D(self):
        D = 0
        for l in self.layers:
            D += self.divergence(l.w_mean, l.w_log_var, l.w_pmean, l.plog_var)\
                + self.divergence(l.b_mean, l.b_log_var, l.b_pmean, l.plog_var)

        return D

    def ELBO(self, x, y, w_nll=1., beta=1., MC_samples=10):

        x_r = torch.cat(MC_samples*[x])
        NLL = self.CE(self.forward(x_r), y.repeat(MC_samples), reduction='sum')

        NLL /= MC_samples
        NLL *= w_nll
        D = self.compute_D()
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
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        t0 = time.time()
        nll = []
        d = []
        acc, ece, mce = [], [], []
        acc20, ece20, mce20 = [], [], []
        KLcol, col = [], []

        for e in range(epochs):

            _nll = 0
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

                _loss = _loss if e >= warm_up else _NLL

                if _loss != _loss:
                    raise NanValues("Aborting training due to nan values")

                _nll += _NLL.item()
                _d += _D.item()

                # Train step
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

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

            _KLcol, _col = self._get_collapsed_posterior()
            KLcol.append(_KLcol)
            col.append(_col)

            nll.append(_nll)
            d.append(_d)

            if v and e % 5 == 4:
                print('On epoch: {:d}, NLL: {:.3e}, '.format(e, nll[-1])
                      + 'D: {:.3e}, collapsed: {:.3f}%'.format(d[-1], col[-1])
                      + ', KL-collapsed: {:.3f}%'.format(KLcol[-1])
                      + ', at time: {:.2f}s'.format(time.time() - t0))

        h = {
            'nll': np.array(nll),
            'divergence': np.array(d),
            'collapsed': np.array(col),
            'KLcollapsed': np.array(KLcol),
            'acc': np.array(acc),
            'ece': np.array(ece),
            'mce': np.array(mce),
            'acc20': np.array(acc20),
            'ece20': np.array(ece20),
            'mce20': np.array(mce20),
        }

        return h

    def fit_prior(self, X, Y, epochs=1000, batch_size=1000, lr=1e-5):

        print('Started point-estimate training..')

        N = X.shape[0]
        n_steps = int(np.ceil(N/batch_size))

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=lr)
        t0 = time.time()
        for e in range(epochs):

            nll = 0
            acc = 0

            # Shuffle data before each epoch
            perm = np.random.permutation(N)
            X = X[perm]
            Y = Y[perm]

            for s in range(n_steps):
                x = X[s*batch_size:min((s+1)*batch_size, N)]
                y = Y[s*batch_size:min((s+1)*batch_size, N)]

                # Use only the means
                logits = self.MAP(x)

                _nll = self.CE(logits, y, reduction='sum')

                _acc = torch.sum(torch.argmax(logits, dim=1)==y).float()

                nll += _nll.item()
                acc += _acc

                # Train step
                optimizer.zero_grad()
                _nll.backward()
                optimizer.step()

            nll /= N
            acc /= N

            if e % 20 == 19:
                print('On epoch: {:d}, NLL: {:.3e}, '.format(e, nll)
                      + 'acc: {:.2f}%, '.format(100.0*acc)
                      + 'at time: {:.2f}s'.format(time.time() - t0))

        # Update prior to match trained weights
        for l in self.layers:
            l.update_prior()
        print('Mean priors updated..')

    def predictive(self,
                   x,
                   dev=None,
                   n_samples=100,
                   step=10,
                   return_list=False,
                   LR=True,
                   prior=False,
                   MAP=False):

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        self.eval()

        if dev is not None:
            curr_dev = next(self.parameters()).device
            self.to(dev)
            x = x.to(dev)

        if MAP:
            z = self.MAP(x)
            preds = self.softmax(z, dim=1).detach()
            if dev is not None:
                self.to(curr_dev)
                preds = preds.to(curr_dev)
            return preds


        preds = 0.0 if not return_list else []
        rem = n_samples
        k = 0

        if prior or (not LR):
            step=1

        while rem:
            k = min(step, rem)
            rem -= k
            x_r = torch.cat(k*[x])
            if prior:
                preds_r = self.softmax(self.prior_forward(x_r).detach(), dim=1)
            else:
                preds_r = self.softmax(self.forward(x_r, LR=LR).detach(), dim=1)
            if not return_list:
                preds += sum(torch.split(preds_r, x.shape[0], dim=0))
            else:
                preds += torch.split(preds_r, x.shape[0], dim=0)
        
        if dev is not None:
            self.to(curr_dev)
            if return_list:
                preds = [pred.to(curr_dev) for pred in preds]
            else:
                preds = preds.to(curr_dev)
        return (preds/n_samples).float() if not return_list \
            else torch.stack(preds).permute(1,0,2)
    
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

    def MAP(self, x):
        Z = 0
        z = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                z = torch.relu(z)

            z = layer.MAP(z)

        if self.lowr:
            zp = torch.zeros((z.shape[0], z.shape[1]+1), device=z.device)
            zp[:, :z.shape[1]] = z
            z = zp

        Z = z

        return Z

    def _get_collapsed_posterior(self):
        # Percentage of collapsed parameters
        with torch.no_grad():
            KL_collaps, collaps, total_params = [0.0]*3
            for l in self.layers:
                KL_collaps += l.get_KLcollapsed_posterior()
                collaps += l.get_collapsed_posterior()
                total_params += l.get_total_params()

            return (100. * KL_collaps/float(total_params),
                    100. * collaps/float(total_params))


    def get_flatten_variances(self):

        variances = np.concatenate([layer.get_flatten_variances() for layer in self.layers])

        return variances