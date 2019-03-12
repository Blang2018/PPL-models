import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import sys

num_chains = int(sys.argv[1])
num_draws = int(sys.argv[2])

oldFaithfulData = pd.read_csv("./datasets/gaussianMixture.data/oldFaithful.csv").values.flatten()
n = oldFaithfulData.size
K = 2

with pm.Model() as gaussianMixtureModel:

    pi = pm.Dirichlet('pi', np.ones(K))

    means = pm.Normal('means', mu=150, sd=100, shape=K)
    sds = pm.Uniform('sds', lower=0.0, upper=100.0, shape=K)

    indicators = pm.Categorical('indicators', p=pi, shape=n)
    observation = pm.Normal('obs', mu=means[indicators], sd=sds[indicators], shape=n, observed=oldFaithfulData)

    trace = pm.sample(1000, chains=1)
    pm.trace_to_dataframe(trace).to_csv("deliverables/gaussianMixture_" + sys.argv[1] + "_" + sys.argv[2] + ".csv")
    # pm.traceplot(trace)
    # plt.show()
