import pandas as pd
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pymc3 as pm, theano.tensor as tt
import sys
import matplotlib.pyplot as plt

num_chains = int(sys.argv[1])
num_draws = int(sys.argv[2])

try:
    df_all = pd.read_csv('data/rugby.csv')
except:
    df_all = pd.read_csv(pm.get_data('rugby.csv'))


def get_tidy_data():
    df_all['diff_non_abs'] = df_all['home_score'] - df_all['away_score']
    # param@values  is the column to be aggregated
    # param@aggfunc is the aggregate function; defaults to np.mean
    # param@index   is the column or "keys" of interest
    # param@columns is the column we group by
    df = df_all[['home_team', 'away_team', 'home_score', 'away_score']]
    teams = df.home_team.unique()
    teams = pd.DataFrame(teams, columns=['team'])
    teams['i'] = teams.index
    # create indices for home team
    df = pd.merge(df, teams, left_on='home_team', right_on='team', how='left')
    # get rid of extra column from the join
    df = df.rename(columns = {'i': 'i_home'}).drop('team', 1)
    # repeat
    df = pd.merge(df, teams, left_on='away_team', right_on='team', how='left')
    df = df.rename(columns = {'i': 'i_away'}).drop('team', 1)

    return df


if __name__ == '__main__':
    df = get_tidy_data()
    obs_h_score = df.home_score.values
    obs_a_score = df.away_score.values
    home_team = df.i_home.values
    away_team = df.i_away.values
    num_teams = max(home_team) + 1

    with pm.Model() as model:
        # home court advantage!
        home = pm.Flat('home')

        sd_atk = pm.HalfStudentT('sd_atk', nu=3, sd=2.5)
        sd_def = pm.HalfStudentT('sd_def', nu=3, sd=2.5)

        # intercept
        intercept = pm.Flat('intercept')

        # team-specific parameters
        # shape parameter for vector of values
        atks_star = pm.Normal('atks_star', mu=0, sd=sd_atk, shape=num_teams)
        defs_star = pm.Normal('defs_star', mu=0, sd=sd_def, shape=num_teams)

        # transformation
        atks = pm.Deterministic('atks', atks_star - tt.mean(atks_star))
        defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))

        # theta as a function of parameters
        home_theta = tt.exp(intercept + home + atks[home_team] + defs[away_team])
        away_theta = tt.exp(intercept        + atks[away_team] + defs[home_team])

        # y | theta ~ Pois(theta)
        home_points = pm.Poisson('home_points', mu=home_theta, observed=obs_h_score)
        away_points = pm.Poisson('away_points', mu=away_theta, observed=obs_a_score)

        trace = pm.sample(draws=num_draws, chains=num_chains)
        pm.trace_to_dataframe(trace).to_csv("deliverables/rugby_" + sys.argv[1] + "_" + sys.argv[2] + ".csv")
        # pm.traceplot(trace)
        # plt.show()



