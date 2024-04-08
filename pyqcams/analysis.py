import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from itertools import groupby
from operator import itemgetter
import os
from pyqcams2.constants import *

fact_sig = (Boh2m*1e2)**2 # bohr^2 to cm^2
fact_k3 = (Boh2m*1e2)/ttos # sigma*v [cm/s]

def opacity(input, GB = True, vib = True, rot = True,  output = None, mode = 'a'):
    '''
    Calculate P(E,b) of a QCT calculation. 
    Has options for state-specific opacities.
    Dissociation data only present if vib == False and rot == False. 
    Input Parameters:
    input, str
        Path to input file with header 
        (e,b,vi,ji,n12,n23,n31,nd,nc,v,vw,j,jw)
    GB, boolean (optional)
        Gaussian Binning option. If False, uses Histogram Binning.
        Default is True.
    vib, boolean (optional)
        Include final vibrational number in analysis.
        Default is True.
    rot, boolean (optional)
        Include final rotational number in analysis.
        Default is True.
    output, str (optional)
        Path to output file, if saving. 
        Default is True.
    mode, str (optional)
        Save mode. Choose from 'w', 'x', or 'a'
        Default is 'w'.  

    '''
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        df = pd.read_csv(input)
    
    cols = ['e','b','vi','ji','n12','n23','n31','nd','nc','v','vw','j','jw']
    df = df.loc[:,cols]

    stats = df.groupby(['e','b','vi','ji']).sum().loc[:,:'nc']
    nTraj = stats[['n12','n23','n31','nd']].sum(axis=1)
    # print(stats.groupby(['e','vi','ji'])['n23','n31'].sum().sum(axis=1)/50)
    # remains = 10000-nTraj[nTraj<10000]
    # print(np.array(remains.values).sum())
    # # print(list(remains.index.get_level_values('b')))
    # print(list(zip(remains.index.get_level_values('b'),remains.values)))

    # State-specific opacity
    weights = df.set_index(['vi','ji'])

    # Gaussian binning
    # G(v,j) = G(v'-vt)*G(j'-jt) from J. Chem. Phys. 133, 164108 (2010)
    weights['w'] = weights['vw']*weights['jw']
    is_AB = (weights['n23'] == 1) | (weights['n31'] == 1) # Check for reaction
    AB = weights[is_AB].groupby(['vi','ji','e','v','j','b']).sum()
    BB = weights[weights['n12'] == 1].groupby(['vi','ji','e','b','v','j']).sum()
    Diss = weights[weights['nd'] == 1].groupby(['vi','ji','e','b']).sum()
    # Net weight should be summed over all v' values, 
    # with the number of dissociation results added.
    net_w = weights.groupby(['vi','ji','e','b']).sum()
    net_w['w']+=net_w['nd']

    # Distributions    
    # Drop NaN values 
    pR = (AB['w']/net_w['w']).fillna(0)
    pR_err = (np.sqrt(AB['w'])/net_w['w']*np.sqrt((net_w['w']-AB['w'])/net_w['w'])).fillna(0)
    pQ = (BB['w']/net_w['w']).fillna(0)
    pQ_err = (np.sqrt(BB['w'])/net_w['w']*np.sqrt((net_w['w']-BB['w'])/net_w['w'])).fillna(0)
    pDiss = (Diss['nd']/net_w['w']).fillna(0)
    pDiss_err = (np.sqrt(Diss['nd'])/net_w['w']*np.sqrt((net_w['w']-Diss['nd'])/net_w['w'])).fillna(0)


    # Histogram binning
    # State-specific
    hR = AB[['n23','n31']].sum(axis=1)/nTraj
    hR_err = np.sqrt(AB[['n23','n31']].sum(axis=1))/nTraj*np.sqrt((nTraj - AB[['n23','n31']].sum(axis=1))/nTraj)
    hQ = BB[['n12']].sum(axis=1)/nTraj
    hQ_err = np.sqrt(BB[['n12']].sum(axis=1))/nTraj*np.sqrt((nTraj-BB[['n12']].sum(axis=1))/nTraj)
    hDiss = Diss[['nd']].sum(axis=1)/nTraj
    hDiss_err = np.sqrt(Diss[['nd']].sum(axis=1))/nTraj*np.sqrt((nTraj-Diss[['nd']].sum(axis=1))/nTraj)


    if vib == True and rot == True:
        if GB == True:
            opacity = pd.DataFrame([pR,pR_err,pQ,pQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            # To add dissociation, reset 'v','j' index, add pDiss
            opacity = opacity.reset_index(level = ['v','j'])
            opacity['pDiss'] = pDiss
            opacity['pDiss_err'] = pDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ']].sum()
            # pnet['pDiss'] = pDiss
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
        else:
            opacity = pd.DataFrame([hR,hR_err,hQ,hQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            opacity = opacity.reset_index(level = ['v','j'])
            # To add dissociation, reset 'v','j' index, add hDiss
            opacity['pDiss'] = hDiss
            opacity['pDiss_err'] = hDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ']].sum()
            # pnet['pDiss'] = hDiss
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
        opacity = opacity.sort_values(by=['v','j'])

    elif vib == True and rot == False:
        if GB == True:
            opacity = pd.DataFrame([pR,pR_err,pQ,pQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            opacity = opacity.groupby(['vi','ji','e','b','v']).sum() # Sum over j
            # To add dissociation, reset 'v' index, add pDiss
            opacity = opacity.reset_index(level = 'v')
            opacity['pDiss'] = pDiss
            opacity['pDiss_err'] = pDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ']].sum()
            # pnet['pDiss'] = pDiss
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
        else:
            opacity = pd.DataFrame([hR,hR_err,hQ,hQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            opacity = opacity.groupby(['vi','ji','e','b','v']).sum() # Sum over j
            # To add dissociation, reset 'v' index, add hDiss
            opacity = opacity.reset_index(level = ['v'])
            opacity['pDiss'] = hDiss
            opacity['pDiss_err'] = hDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ']].sum()
            # pnet['pDiss'] = hDiss
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
        opacity = opacity.sort_values(by=['v'])
    elif vib == False and rot == True:
        if GB == True:
            opacity = pd.DataFrame([pR,pR_err,pQ,pQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            opacity = opacity.groupby(['vi','ji','e','b','j']).sum() # Sum over v
            # To add dissociation, reset 'v' index, add pDiss
            opacity = opacity.reset_index(level = 'j')
            opacity['pDiss'] = pDiss
            opacity['pDiss_err'] = pDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ']].sum()
            # pnet['pDiss'] = pDiss
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
        else:
            opacity = pd.DataFrame([hR,hR_err,hQ,hQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            opacity = opacity.groupby(['vi','ji','e','b','j']).sum() # Sum over v, j
            # To add dissociation, reset 'j' index, add hDiss
            opacity = opacity.reset_index(level = 'j')
            opacity['pDiss'] = pDiss
            opacity['pDiss_err'] = pDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ']].sum()
            # pnet['pDiss'] = pDiss
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
        opacity = opacity.sort_values(by=['j'])

    elif vib == False and rot == False:
        if GB == True:
            opacity = pd.DataFrame([pR,pR_err,pQ,pQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            opacity = opacity.groupby(['vi','ji','e','b']).sum() # Sum over v, j
            opacity['pDiss'] = pDiss
            opacity['pDiss_err'] = pDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ','pDiss']].sum()
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
        else:
            opacity = pd.DataFrame([hR,hR_err,hQ,hQ_err], index=['pR','pR_err','pQ','pQ_err']).T
            opacity = opacity.groupby(['vi','ji','e','b']).sum()
            opacity['pDiss'] = hDiss
            opacity['pDiss_err'] = hDiss_err
            # Check net probability = 1
            # pnet = opacity.groupby(['vi','ji','e','b'])[['pR','pQ','pDiss']].sum()
            # print(pnet[['pR','pQ','pDiss']].sum(axis=1))
    
    opacity = opacity.fillna(0)

    if output is not None:
        opacity.to_csv(f'{output}', mode = mode, header = os.path.isfile(output) == False or os.path.getsize(output) == 0)

    return opacity
    
def crossSection(input, GB = True, vib = True, rot = True,  output = None, mode = 'a'):
    '''
    Calculate cross section, sigma(E), of a QCT calculation. 
    Has options for state-specific cross sections.

    Input Parameters:
    input, str or pandas DataFrame
        Path to input file with header or opacity DataFrame
        (e,b,vi,ji,n12,n23,n31,nd,nc,v,vw,j,jw)
    GB, boolean (optional)
        Gaussian Binning option. If False, uses Histogram Binning.
        Default is True.
    vib, boolean (optional)
        Include final vibrational number in analysis.
        Default is True.
    rot, boolean (optional)
        Include final rotational number in analysis.
        Default is True.
    output, str (optional)
        Path to output file, if saving. 
        Default is True.
    mode, str (optional)
        Save mode. Choose from 'w', 'x', or 'a'
        Default is 'w'.  

    '''
    # If input is not opacity dataframe
    if isinstance(input,pd.DataFrame):
        opac = input.copy()
    else:    
        opac = opacity(input,  GB = GB, vib = vib, rot = rot).copy() # Copy input dataframe
    opac = opac.reset_index(level='b').sort_values(by=['b']) # Sort b-values for integration

    def integ(integrand,x):
        return 2*np.pi*integrate.trapz(integrand,x=x)
    
    if 'v' not in opac.columns and 'j' not in opac.columns:
        sig_R = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pR*g.b, x=g.b))*fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pR_err*g.b, x=g.b))*fact_sig
        sig_Q = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pQ*g.b, x=g.b))*fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pQ_err*g.b, x=g.b))*fact_sig
        sigma = pd.DataFrame([sig_R,sig_Rerr,sig_Q,sig_Qerr],
                          index = ['sig_R','sig_Rerr','sig_Q','sig_Qerr']).T
        

    elif 'v' in opac.columns and 'j' not in opac.columns:
        sig_R = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pR*g.b, x=g.b))*fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pR_err*g.b, x=g.b))*fact_sig
        sig_Q = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pQ*g.b, x=g.b))*fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e','v']).apply(lambda g: integ(g.pQ_err*g.b, x=g.b))*fact_sig
        sigma = pd.DataFrame([sig_R,sig_Rerr,sig_Q,sig_Qerr],
                          index = ['sig_R','sig_Rerr','sig_Q','sig_Qerr']).T
        sigma = sigma.reset_index(level=['v'])

    elif 'v' not in opac.columns and 'j' in opac.columns:
        sig_R = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pR*g.b, x=g.b))*fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pR_err*g.b, x=g.b))*fact_sig
        sig_Q = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pQ*g.b, x=g.b))*fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e','j']).apply(lambda g: integ(g.pQ_err*g.b, x=g.b))*fact_sig
        sigma = pd.DataFrame([sig_R,sig_Rerr,sig_Q,sig_Qerr],
                          index = ['sig_R','sig_Rerr','sig_Q','sig_Qerr']).T
        sigma = sigma.reset_index(level=['j'])

    elif 'v' in opac.columns and 'j' in opac.columns:
        sig_R = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pR*g.b, x=g.b))*fact_sig
        sig_Rerr = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pR_err*g.b, x=g.b))*fact_sig
        sig_Q = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pQ*g.b, x=g.b))*fact_sig
        sig_Qerr = opac.groupby(['vi','ji','e','v','j']).apply(lambda g: integ(g.pQ_err*g.b, x=g.b))*fact_sig
        sigma = pd.DataFrame([sig_R,sig_Rerr,sig_Q,sig_Qerr],
                          index = ['sig_R','sig_Rerr','sig_Q','sig_Qerr']).T
        sigma = sigma.reset_index(level=['v','j'])
 
    # Dissociation is not state-specific
    sigma['sig_Diss'] = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pDiss*g.b, x=g.b))*fact_sig
    sigma['sig_Disserr'] = opac.groupby(['vi','ji','e']).apply(lambda g: integ(g.pDiss_err*g.b, x=g.b))*fact_sig

    if output is not None:
        sigma.to_csv(f'{output}', mode = mode, header = os.path.isfile(output) == False or os.path.getsize(output) == 0)

    return sigma


def rate(input, mu, GB = True, vib = True, rot = True,  output = None, mode = 'a'):
    k = crossSection(input, GB = GB, vib = vib, rot = rot).copy()
    k['P0'] = np.sqrt(k.index.get_level_values(level = 'e')*2*mu*K2Har)
    k[['sig_R','sig_Rerr','sig_Q','sig_Qerr','sig_Diss','sig_Disserr']].multiply(k['P0']/mu*fact_k3, axis='index')
    k = k.rename(columns = {'sig_R':'k_R','sig_Rerr':'k_Rerr',
                           'sig_Q':'k_Q','sig_Qerr':'k_Qerr',
                           'sig_Diss':'k_Diss','sig_Disserr':'k_Disserr'})
    k = k.drop(columns = 'P0')

    if output is not None:
        k.to_csv(f'{output}', mode = mode,header = os.path.isfile(output) == False or os.path.getsize(output) == 0)

    return k

if __name__ == '__main__':
    # from inputs import *
    # m1,m2 = input_dict['mol_AB']['mi'],input_dict['mol_AB']['mj']
    # m3 = input_dict['mol_CA']['mj']
    # mu312 = m3*(m1+m2)/(m1+m2+m3)
    # opac = opacity('examples/RbBa+/newerresults/v189_long.txt')
    i=187
    opac = opacity(f'examples/RbBa+/newerresults/v{i}_long.txt', GB = False, vib = True, rot = False)
    opacv = opac[opac['v'] == i-1].drop('v',axis=1) # Don't include v=vi
    # Surgery
    opacD = opac[opac['v'] != i-1].drop('v',axis=1)[['pDiss','pDiss_err']].drop_duplicates() # Dissociation repeats for every v
    probs = opac[opac['v'] != i-1].drop(['pR_err','pQ_err','pDiss','pDiss_err'],axis=1).groupby(['vi','ji','e','b']).sum().drop('v',axis=1) # Drop diss; Sum over pR,pQ; drop v
    errs =  opac[opac['v'] != i-1].drop(['pR','pQ','pDiss','pDiss_err'],axis=1).groupby(['vi','ji','e','b']).transform(lambda x: np.linalg.norm(x)).drop('v',axis=1).drop_duplicates() 
    # print(opacD)
    # print(probs)
    # print(errs)
    print(pd.concat([probs,errs,opacD],axis=1).fillna(0))
    # opac = pd.concat([opac,opacD],axis=1).fillna(0)    
    # print(opac)
    # sig = crossSection(opac)
    # print(sig)
    # k = rate('long.txt',mu312=mu312,GB = False, vib = True, rot = True)
    # print(k[k['k_R']!=0])

    
