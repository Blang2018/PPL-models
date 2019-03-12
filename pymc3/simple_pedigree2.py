import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df_pedigree = pd.read_csv('../data/schork-guo-fig4-missing2.csv')
    dis0 = df_pedigree["disease"][df_pedigree["genotype"] == 0]
    dis1 = df_pedigree["disease"][df_pedigree["genotype"] == 1]
    dis2 = df_pedigree["disease"][df_pedigree["genotype"] == 2]
    al1 = df_pedigree['al1']
    al2 = df_pedigree['al2']
    founders = np.ma.masked_invalid(df_pedigree["genotype"][df_pedigree["par1"] == -1])
    members = np.ma.masked_invalid(df_pedigree["genotype"][df_pedigree["par1"] >= 0])
    members_par = [df_pedigree["par1"][(df_pedigree["par1"] >= 0)], df_pedigree["par2"][(df_pedigree["par1"] >= 0)]]

    al00 = np.array([al1[id] for id in members_par[0].values])
    al01 = np.array([al2[id] for id in members_par[0].values])
    al10 = np.array([al1[id] for id in members_par[1].values])
    al11 = np.array([al2[id] for id in members_par[1].values])
    cat_param = np.zeros(shape=(members.size, 3))
    for i in np.arange(0, members.size):
        if (np.isnan(al00[i])):
            cat_param[i][0] = np.ma.masked
            cat_param[i][1] = np.ma.masked
            cat_param[i][2] = np.ma.masked
            continue
        cat_param[i][int(al00[i] + al10[i])] += 1
        cat_param[i][int(al00[i] + al11[i])] += 1
        cat_param[i][int(al01[i] + al10[i])] += 1
        cat_param[i][int(al01[i] + al11[i])] += 1
    cat_param = np.ma.masked_invalid(cat_param)/4.0
    print(cat_param)

    with pm.Model() as model:
        # pi       = pm.Uniform('pi', lower=0, upper=1, shape=2)
        # disease0 = pm.Bernoulli('dis0', p=pi[0], observed=dis0)
        # disease1 = pm.Bernoulli('dis1', p=pi[1], observed=dis1)
        # disease2 = pm.Bernoulli('dis2', p=0.0, observed=dis2)
        #
        # p_founders = pm.Dirichlet("p_founders", a=np.ones(3))
        # g_founders = pm.Categorical("g_founders", p=p_founders, observed=founders)

        g_members = pm.Categorical("g_members", p=cat_param, shape=members.size, observed=members)

        trace = pm.sample(tune=2000)
        pm.traceplot(trace, varnames=['pi'])
        plt.show()







