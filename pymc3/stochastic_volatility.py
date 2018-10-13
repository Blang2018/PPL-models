import numpy as np
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk





if __name__ == "__main__":
    returns = np.genfromtxt(pm.get_data("SP500.csv"))

    with pm.Model() as model:
        step_size = pm.Exponential('sigma', 50.)
        s = GaussianRandomWalk('s', sd=step_size, shape=len(returns))
        nu = pm.Exponential('nu', .1)
        r = pm.StudentT('r', nu=nu, lam=pm.math.exp(-2 * s), observed=returns)
        trace = pm.sample(tune=5000, nuts_kwargs=dict(target_accept=.9))

        pm.traceplot(trace, varnames=['sigma', 'nu'])







