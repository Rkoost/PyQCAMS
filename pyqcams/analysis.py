import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import integrate
import constants
import pandas as pd
import json

def input(input_file):
    # Input data
    with open(input_file) as f:
        data = json.load(f)

    m1,m2,m3 = data['masses'].values()
    return m1, m2, m3

def load_df(input_file, data_file):
    m1, m2, m3 = input(input_file)
    mu12 = m1*m2/(m1+m2) # H2
    mu31 = m1*m3/(m1+m3) # df
    mu32 = m2*m3/(m2+m3) # df
    mu123 = m3*(m1+m2)/(m1+m2+m3)

    
def distribution(input_file, data_file, prod = 'q'):
    '''
    df, Pandas DataFrame

    p, string
        Choose which product to plot from 
        ['q', 'r1', 'r2', 'diss', 'comp']. 

    '''
    m1, m2, m3 = input(input_file)
    mu12 = m1*m2/(m1+m2) # H2
    mu31 = m1*m3/(m1+m3) # df
    mu32 = m2*m3/(m2+m3) # df
    mu123 = m3*(m1+m2)/(m1+m2+m3)

    df = pd.read_csv(data_file)
    df = df[df['v'] >= 0]
    stats = df.loc[:,'e':'diss'].groupby(['e','b']).sum() # Counts
    weights = df.drop(df.loc[:,'d_i':], axis=1) # counts with weights
    nt = stats.sum(axis=1) # Total number of trajectories for a given (E,b)


    prod_w = weights[weights[prod] == 1].groupby(['e','b','v']).sum()
    net_w = weights.groupby(['e','b']).sum() # Sum all weights
    net_w['w'] += net_w['diss'] # Add dissociation weights of 1

    prod_df = df[df[prod] == 1].groupby(['e','b','v']).sum()
    prod_df['p'] = prod_w['w']/net_w['w'] # Probability of reaction P(E,b,v)
    prod_df['p_err'] = np.sqrt(prod_w['w'])/net_w['w']*np.sqrt((net_w['w']-prod_w['w'])/net_w['w'])
    ### Calculate cross section & rates ### 
    prod_df = prod_df.reset_index() # Drop dependence on b to integrate
    prod_df.set_index(['e','v'], inplace=True)
    prod_df['s'] = prod_df.groupby(['e','v']).apply(lambda g: 8*np.pi**2*integrate.trapz(g.p*g.b, x=g.b))
    prod_df['s_err'] = prod_df.groupby(['e','v']).apply(lambda g: 8*np.pi**2*integrate.trapz(g.p_err*g.b, x=g.b))
    prod_df['k'] = np.sqrt(2*3/2*constants.kb*prod_df['s'].index.get_level_values(level = 'e')/mu123)*prod_df['s']*constants.autocm**3/constants.ttos
    prod_df['k_err'] = np.sqrt(2*3/2*constants.kb*prod_df['s_err'].index.get_level_values(level = 'e')/mu123)*prod_df['s_err']*constants.autocm**3/constants.ttos
    
    return prod_df

def kdist_plt(df, log = True):
    df = df.reset_index()
    evals = df.e.unique() # List of energy values
    ax = df[df['e']==evals[0]].plot.scatter(x='v',y=f'k', yerr = f'k_err', marker ='.', label = f'{evals[0]} K')
    cmap = mpl.colormaps['viridis'] # Color map
    colors = cmap(np.linspace(0, 1, len(evals))) 
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if len(evals > 0): 
        for idx,e in enumerate(evals[1:]): 
            idx += 1
            df[df['e']==e].plot.scatter(x='v',y=f'k', yerr = f'k_err', marker ='.', label = f'{e} K', c=colors[idx], ax=ax)
    if log == True:
        ax.set_yscale('log')
    ax.tick_params(axis = 'x', direction = 'in')
    ax.tick_params(axis = 'y', direction = 'in', which = 'both')
    plt.xticks(np.arange(0,int(max(df.v.unique()))+2, 2))
    plt.ylabel(f'$k_r (cm^3/s)$')
    plt.title('Rate Distribution ')
    plt.show()

def pdist_plt(df, b):
    df = df.reset_index()
    df = df[df['b'] == b]
    evals = df.e.unique() # List of energy values
    ax = df[df['e']==evals[0]].plot(x='v',y=f'p', yerr = f'p_err', marker ='.', label = f'{evals[0]} K')
    cmap = mpl.colormaps['viridis'] # Color map
    colors = cmap(np.linspace(0, 1, len(evals))) 
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if len(evals > 0): 
        for idx,e in enumerate(evals[1:]): 
            idx += 1
            df[df['e']==e].plot(x='v',y=f'p', yerr = f'p_err', marker ='.', label = f'{e} K', c=colors[idx], ax=ax)
    ax.tick_params(axis = 'x', direction = 'in')
    ax.tick_params(axis = 'y', direction = 'in', which = 'both')
    plt.xticks(np.arange(0,int(max(df.v.unique()))+2, 2))
    plt.ylabel(f'$P (E_c)$')
    plt.title(f'Probability Distribution (b = {b})')
    plt.show()

def rates(input_file, data_files, prod = 'q'):
    ### Concat all input files into a dataframe, then p
    m1, m2, m3 = input(input_file)
    mu12 = m1*m2/(m1+m2) 
    mu31 = m1*m3/(m1+m3) 
    mu32 = m2*m3/(m2+m3) 
    mu123 = m3*(m1+m2)/(m1+m2+m3)
    
    # DATAFRAMES
    df = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)
    df = df[df['v'] >= 0]
    df = df.set_index('n_i')
    stats = df.loc[:,'e':'diss'].groupby(['e','b']).sum() # Counts
    weights = df.drop(df.loc[:,'d_i':], axis=1) # counts with weights
    nt = stats.sum(axis=1) # Total number of trajectories for a given (E,b)

    # WEIGHTS
    prod_w = weights[weights[prod] == 1].groupby(['n_i','e','b','v']).sum()
    net_w = weights.groupby(['e','b']).sum() # Sum all weights
    net_w['w'] += net_w['diss'] # Add dissociation weights of 1

    # DISTRBUTIONS
    prod_df = df[df[prod] == 1].groupby(['n_i','e','b','v']).sum()
    prod_df['p'] = prod_w['w']/net_w['w'] # Probability of reaction P(E,b,v)
    prod_df['p_err'] = np.sqrt(prod_w['w'])/net_w['w']*np.sqrt((net_w['w']-prod_w['w'])/net_w['w'])
    ### Calculate cross section & rates ### 
    prod_df = prod_df.reset_index(level = 2) # Drop dependence on b to integrate
    # prod_df.set_index(['e','v'], inplace=True)
    prod_df['s'] = prod_df.groupby(['n_i','e','v']).apply(lambda g: 8*np.pi**2*integrate.trapz(g.p*g.b, x=g.b))
    prod_df['s_err'] = prod_df.groupby(['n_i','e','v']).apply(lambda g: 8*np.pi**2*integrate.trapz(g.p_err*g.b, x=g.b))
    prod_df['k'] = np.sqrt(2*3/2*constants.kb*prod_df['s'].index.get_level_values(level = 'e')/mu123)*prod_df['s']*constants.autocm**3/constants.ttos
    prod_df['k_err'] = np.sqrt(2*3/2*constants.kb*prod_df['s_err'].index.get_level_values(level = 'e')/mu123)*prod_df['s_err']*constants.autocm**3/constants.ttos
    
    #Calculate P(E,b), sigma(E), k(E)
    df_eb = prod_df.reset_index().groupby(['n_i','e','b']).sum()
    df_eb = df_eb.reset_index(level = 2) # Drop dependence on b to integrate
    df_eb['s(e)'] = df_eb.groupby(['n_i','e']).apply(lambda g: 8*np.pi**2*integrate.trapz(g.p*g.b, x=g.b))
    df_eb['s_err(e)'] = df_eb.groupby(['n_i','e']).apply(lambda g: 8*np.pi**2*integrate.trapz(g.p_err*g.b, x=g.b))
    df_eb['k(e)'] = np.sqrt(2*3/2*constants.kb*df_eb['s(e)'].index.get_level_values(level = 'e')/mu123)*df_eb['s(e)']*constants.autocm**3/constants.ttos
    df_eb['k_err(e)'] = np.sqrt(2*3/2*constants.kb*df_eb['s_err(e)'].index.get_level_values(level = 'e')/mu123)*df_eb['s_err(e)']*constants.autocm**3/constants.ttos
    df_eb['s(e)'] = df_eb['s(e)'].drop_duplicates()
    return prod_df, df_eb

if __name__ == '__main__':
    # h2_dist = distribution('inputs.json', 'new_data/v0j0.csv', prod = 'h2')
    # # # print(h2['n_i'])
    # kdist_plt(h2, log = False)
    # pdist_plt(h2, b = 0)
    h2_dist, h2_rate = rates('inputs.json', ['new_data/v0j0.csv'], prod = 'h2')
    pdist_plt(h2_dist, b = 0)