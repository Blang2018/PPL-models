import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
from pymc3.distributions.timeseries import GaussianRandomWalk
import sys

num_chains = int(sys.argv[1])
num_draws = int(sys.argv[2])
DATA_PATH = sys.argv[-1]

returns = pd.read_csv(DATA_PATH + "datasets/sv.data/SP500.csv")["change"].values.flatten()

with pm.Model() as model:
    sigma = pm.Exponential('sigma', 50.)
    s = GaussianRandomWalk('s', sd=sigma, shape=len(returns))
    nu = pm.Exponential('nu', .1)
    r = pm.StudentT('r', nu=nu, lam=pm.math.exp(-2 * s), observed=returns)

    trace = pm.sample(draws=num_draws, chains=num_chains)
    pm.trace_to_dataframe(trace).to_csv("deliverables/sv_" + sys.argv[1] + "_" + sys.argv[2] + ".csv")
    # pm.traceplot(trace, varnames=['sigma', 'nu'])
    # plt.show()








