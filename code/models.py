import time

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax, cross_entropy, softplus, tanh


class NanValues(Exception):
    """"Custom exception to signal failed training"""


class TempScaling(nn.Module):

    def __init__(self):
        super(TempScaling, self).__init__()

        # Init temperature
        self.T = nn.Parameter(torch.Tensor([1.0]))

    def fit(self, X, y, v=False):

        optim = torch.optim.SGD(self.parameters(), lr=1e-1)

        self.train()

        _T = 1.0
        e=0
        t0 = time.time()
        while True:
            loss = cross_entropy(self.forward(X), y)

            if loss != loss:
                raise NanValues("Aborting training due to nan values")

            optim.zero_grad()
            loss.backward()
            optim.step()

            if v and e % 10 == 4:
                print('On epoch: {:d}, loss: {:.3e}, '.format(e, loss.item())
                     + 'Temp: {:.3f}, '.format(self.T.item())
                      + ', at time: {:.2f}s'.format(time.time() - t0), end="\r")
            e += 1

            if np.abs(self.T.item()-_T)<1e-7:
                break
            else:
                _T = self.T.item()

        return self.forward(X)

    def forward(self, x):
        return x/self.T

    def predictive(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        return softmax(self.forward(x), dim=-1)


class AdaptiveTempScalingv3(TempScaling):

    def __init__(self, dim):
        super(AdaptiveTempScalingv3, self).__init__()

        # Init params
        self.b = nn.Parameter(torch.Tensor([1.0]))
        self.W = nn.Parameter(torch.randn(dim)/(100*dim))

        self.dim = dim

    def fit(self, X, Y, epochs=10000, batch_size=None, lr=1e-5, v=False, weight_decay=0.1):

        # Compute optimum T first
        optim = torch.optim.SGD([self.b], lr=1e-1)

        self.train()

        _T = 1.0
        e=0
        t0 = time.time()
        if v:
            print('Finding optimum Temperature')
        while True:
            loss = cross_entropy(X/self.b, Y)

            if loss != loss:
                raise NanValues("Aborting training due to nan values")

            optim.zero_grad()
            loss.backward()
            optim.step()

            if v and e % 10 == 4:
                print('On epoch: {:d}, loss: {:.3e}, '.format(e, loss.item())
                     + 'Temp: {:.3f}, '.format(self.b.item())
                      + ', at time: {:.2f}s'.format(time.time() - t0), end="\r")
            e += 1

            if np.abs(self.b.item()-_T)<1e-7:
                break
            else:
                _T = self.b.item()

        optim = torch.optim.SGD([self.W], lr=lr, weight_decay=weight_decay)

        self.train()

        N = X.shape[0]
        if batch_size is None:
            batch_size=N
        n_steps = int(np.ceil(N/batch_size))

        nll = 0

        e=0
        t0 = time.time()
        if v:
            print('Adapting Weight vector')
            print('\n')
        while e<epochs:

            # Shuffle data before each epoch
            perm = np.random.permutation(N)
            X = X[perm]
            Y = Y[perm]

            for s in range(n_steps):
                x = X[s*batch_size:min((s+1)*batch_size, N)]
                y = Y[s*batch_size:min((s+1)*batch_size, N)]

                logits = self.forward(x)

                _nll = cross_entropy(logits, y, reduction='sum')

                if _nll != _nll:
                    raise NanValues("Aborting training due to nan values")

                # Train step
                optim.zero_grad()
                _nll.backward()
                optim.step()

                nll += _nll.item()

            nll /= N
            if v and e % 10 == 4:
                print('On epoch: {:d}, loss: {:.3e}, '.format(e, nll)
                      + 'at time: {:.2f}s'.format(time.time() - t0), end="\r")
            e += 1
        print('\n')

        return self.forward(X)

    def forward(self, x):
        # T = nn.functional.relu(x @ self.W) + self.b
        T = (tanh((x @ self.W)/self.dim) + 1) * self.b
        return x/T.view(-1, 1)

    def get_T(self, x):
        T = (tanh((x @ self.W)/self.dim) + 1) * self.b
        return T.detach().numpy()

    def predictive(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        return softmax(self.forward(x), dim=-1)


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
        return softmax(z, dim=1)
