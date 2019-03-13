import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import sys

num_chains = int(sys.argv[1])
num_draws = int(sys.argv[2])
sampling_seed = (int(sys.argv[3])+1)^10
DATA_PATH = sys.argv[-1]

oldFaithfulData = pd.read_csv(DATA_PATH + "datasets/gaussianMixture.data/oldFaithful.csv").values.flatten()
n = oldFaithfulData.size
K = 2

with pm.Model() as gaussianMixtureModel:

    pi = pm.Dirichlet('pi', np.ones(K))

    means = pm.Normal('means', mu=150, sd=100, shape=K)
    sds = pm.Uniform('sds', lower=0.0, upper=100.0, shape=K)

    indicators = pm.Categorical('indicators', p=pi, shape=n)
    observation = pm.Normal('obs', mu=means[indicators], sd=sds[indicators], shape=n, observed=oldFaithfulData)

    trace = pm.sample(draws=num_draws, chains=num_chains, random_seed=sampling_seed)
    results = pm.trace_to_dataframe(trace)
    for header in list(results):
        output = results.loc[:,header]
        output.to_csv("gaussianMixture_" + header + "_" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] +
                      ".csv", index_label="sample", header=["value"])
    # pm.traceplot(trace)
    # plt.show()
