import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import sys

num_chains = int(sys.argv[1])
num_draws = int(sys.argv[2])
sampling_seed = (int(sys.argv[3])+1)^10
DATA_PATH = sys.argv[-1]

texting_data = pd.read_csv(DATA_PATH + "datasets/changePoint.data/texting-data.csv").values.flatten()
n = texting_data.size

with pm.Model() as cp_model:

    changePoint = pm.DiscreteUniform('change_point', lower=0, upper=n)

    lambda1 = pm.Exponential(name="lambda1", lam=1.0/15.0)
    lambda2 = pm.Exponential(name="lambda2", lam=1.0/15.0)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(changePoint >= np.arange(0, n), lambda1, lambda2)

    counts = pm.Poisson('counts', rate, observed=texting_data)

    trace = pm.sample(draws=num_draws, chains=num_chains, random_seed=sampling_seed)
    results = pm.trace_to_dataframe(trace)
    for header in list(results):
        output = results.loc[:,header]
        output.to_csv("changePoint_" + header + "_" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] +
                      ".csv", index_label="sample", header=["value"])
    # pm.traceplot(trace)
    # plt.show()

