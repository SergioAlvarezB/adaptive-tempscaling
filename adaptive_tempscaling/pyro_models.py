import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive


class TempScaling():

    def __init__(self, pmean=0.0, plog_var=0.25) -> None:
        super().__init__()
        self.pmean = pmean
        self.plog_var = plog_var
        self.model = self._get_model(pmean, plog_var)

    def _get_model(self, pmean, plog_var) -> callable:
        def mcmc_TS(logits, targets):
            invT = pyro.sample("invT", dist.LogNormal(pmean, plog_var))

            calibrated_logits = pyro.deterministic('calibrated_logits', logits * invT)
            with pyro.plate('data', len(logits)):
                return pyro.sample("obs", dist.Categorical(logits=calibrated_logits), obs=targets)
            
        return mcmc_TS
    
    def fit(self, X_val, Y_val, num_samples=500):
        self.nuts_kernel = NUTS(self.model)
        self.mcmc = MCMC(self.nuts_kernel, num_samples=num_samples)
        self.mcmc.run(X_val, Y_val)


    def predict(self, X_test, samples=None, reduce=True):
        if samples is None:
            samples = self.mcmc.get_samples()
        self.predictive = Predictive(self.model, samples)(X_test, None)

        if reduce:
            return torch.mean(self.predictive['calibrated_logits'], dim=0)
        return self.predictive['calibrated_logits']