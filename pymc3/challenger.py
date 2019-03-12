import numpy as np
import pymc3 as pm
import pandas as pd
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":

    num_chains = int(sys.argv[1])
    num_draws = int(sys.argv[2])
    DATA_PATH = sys.argv[-1]

    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))


    incidents_data = np.ma.masked_invalid(pd.read_csv(DATA_PATH + "datasets/challenger.data/incidents.csv").values).flatten()
    temperatures = pd.read_csv(DATA_PATH + "datasets/challenger.data/temperatures.csv").values.flatten()

    with pm.Model() as challenger_model:
        intercept = pm.Normal('intercept', mu=0.0, sd=10.0)
        slope = pm.Normal('slope', mu=0.0, sd=10.0)

        incidents = pm.Bernoulli('incidents',
                                 p=logistic(intercept + slope * temperatures),
                                 shape=incidents_data.size,
                                 observed=incidents_data)

        trace = pm.sample(draws=num_draws, chains=num_chains)
        pm.trace_to_dataframe(trace).to_csv("deliverables/challenger_" + sys.argv[1] + "_" + sys.argv[2] + ".csv")
        # pm.plot_posterior(trace)
        # plt.show()


