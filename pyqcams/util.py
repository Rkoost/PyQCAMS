import os
import warnings
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
from pyqcams import constants

def gaus(vp , vt, s=0.1):
            ''' Gaussian function used for Gaussian Binning of final vibrational states.
            Inputs:
            vp, float
                vPrime output, some number between integers
            vt, int
                target vibrational quantum number. 
            s, float
                sigma or FWHM of Gaussian

            Returns:
            w, float 
                W(vp,vt) which associates a weight to a vibrational product state
            '''
            w = np.exp(-np.abs(vp-vt)**2/2/s**2)
            w *= 1/np.sqrt(2*np.pi)/s
            return w
            

def bound(v,dv, j, mu, re):
    ''' Finds the new boundary for higher j values
    Purpose:
        For higher j values, the "bound" condition changes
        from E < 0 to E < j*(j+1)/2/mu/ro**2
        This function finds r_o at which this boundary is created.
    '''
    if j == 0:
        bdry = 0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    
            vp_eff = lambda r: dv(r) - j*(j+1)/mu/r**3
            v_eff = lambda r: v(r) + j*(j+1)/2/mu/r**2
            ro = fsolve(vp_eff, re+3)  # 3 a_0 away from the equlibrium seems to work (for Morse)
            bdry = v_eff(ro)
    return bdry


def get_results(a):
    '''Get long output from trajectory
    a, object
        trajectory object containing relevant attributes
    '''
    results = {'e': a.e0/constants.cEK2H, # energy
            'b': a.b, # impact param   
            'q': a.count[0], # quench
            'r1': a.count[1], # reaction 1
            'r2': a.count[2], # reaction 2
            'diss': a.count[3],  # dissociation
            'comp': a.count[4], # complex
            'v': a.f_state[0], # final vib num
            'w': a.f_state[1], # weight
            'j': a.f_state[2], # final j
            'd_i': a.d, # initial distance
            'theta': a.ang[0], # initial angles
            'phi': a.ang[1], 
            'eta': a.ang[2], 
            'n_i': a.n_vib, # initial vib num
            'j_i': a.j, # initial j 
            'rho1x': a.r[0][-1], # final positions
            'rho1y': a.r[1][-1],
            'rho1z': a.r[2][-1],
            'rho2x': a.r[0][-1],
            'rho2y': a.r[1][-1],
            'rho2z': a.r[2][-1],
            'p1x': a.f_p[0], # final momentum 
            'p1y': a.f_p[1],
            'p1z': a.f_p[2],
            'p2x': a.f_p[3],
            'p2y': a.f_p[4],
            'p2z': a.f_p[5],
            'tf': a.t[-1]} # final time
    
    return results

def numToV(file):
    '''
    Convert pdf table to potential function and first derivative.
    Assumes positions in column 0, energy in column 1.

    Inputs:

    file, str
        path to pdf file

    Outputs:
    num_x, list
        x values of potential
    num_V, function
        interpolated potential
    num_dV, function
        interpolated potential derivative
    num_re, float
        equilibrium point (min of potential)
    '''
    types = ['.csv','.txt','.dat']
    split = os.path.splitext(f'{file}')
    filetype = split[1]
    if filetype in types:
        try:
            # comma separated
            df = pd.read_csv(f'{file}',header = None)
            num_x = np.array([float(i) for i in df[0][:].values.tolist()])
            num_y = np.array([float(i) for i in df[1][:].values.tolist()])
        except:
            try:
                # tab separated
                df = pd.read_csv(f'{file}',header = None, sep = '\s+')
                num_x = np.array([float(i) for i in df[0][:].values.tolist()])
                num_y = np.array([float(i) for i in df[1][:].values.tolist()])
            except:
                print('Please use either comma or tab separated data.')
                quit()
    else: 
        print('Please specify filetype. Choose from "csv", "dat", or "txt".')
    num_y -= num_y[-1] # Shift values to make curve approach 0 by setting final point to 0
    num_V = InterpolatedUnivariateSpline(num_x,num_y, k = 4, ext = 'raise') # Use k = 4 to find roots of derivative
    num_dV = num_V.derivative()
    cr_pts = num_dV.roots()
    cr_pts = np.append(cr_pts, (num_x[0], num_x[-1]))  # also check the endpoints of the interval
    cr_vals = num_V(cr_pts)
    min_index = np.argmin(cr_vals)
    num_re = cr_pts[min_index] # Equilibrium distance
    return num_x, num_V, num_dV, num_re

