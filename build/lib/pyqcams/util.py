import numpy as np
import warnings
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import multiprocess as mp 
import pandas as pd
from . import pymar
from .constants import *
import os 

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
    results = {'e': a.e0/cEK2H,
            'b': a.b,
            'q': a.count[0],
            'r1': a.count[1],
            'r2': a.count[2],
            'diss': a.count[3],
            'comp': a.count[4],
            'v': a.f_state[0],
            'w': a.f_state[1],
            'j': a.f_state[2],
            'd_i': a.d,
            'theta': a.ang[0], 
            'phi': a.ang[1], 
            'eta': a.ang[2], 
            'n_i': a.n_vib,
            'j_i': a.j,
            'rho1x': a.r[0][-1],
            'rho1y': a.r[1][-1],
            'rho1z': a.r[2][-1],
            'rho2x': a.r[0][-1],
            'rho2y': a.r[1][-1],
            'rho2z': a.r[2][-1],
            'p1x': a.f_p[0],
            'p1y': a.f_p[1],
            'p1z': a.f_p[2],
            'tf': a.t[-1]}
    
    return results

def save_long(n_traj,cpus,calc,out_file):
    '''
    Runs parallel trajectories.
    Saves a long version of the data; one line per trajectory.
    Stores all input & output data.

    n_traj, int
        number of trajectories
    cpus, int
        number of cpus for parallel calculation
    calc, dict
        output from pymar.start() function
    out_file, string
        output file name
    '''
    with mp.Pool(processes = cpus) as p:
        event = [p.apply_async(pymar.main, kwds = (calc)) for i in range(n_traj)]
        for res in event:
            df = pd.DataFrame([res.get()])
            df.to_csv(out_file, mode = 'a', index = False, 
                    header = os.path.isfile(out_file) == False or os.path.getsize(out_file) == 0)
    return

def save_short(n_traj,cpus,calc,out_file):
    '''
    Runs parallel trajectories.
    Saves a short version of the data; sums up the counts so one line per (E,b) set.
    Use this if input data & distributions are not needed.

    Set cpus = 1 for serial calculation.

    n_traj, int
        number of trajectories
    cpus, int
        number of cpus for parallel calculation
    calc, dict
        output from pymar.start() function
    out_file, string
        output file name
    '''
    result = []
    with mp.Pool(processes = cpus) as p:
        event = [p.apply_async(pymar.main, kwds = (calc)) for i in range(n_traj)]
        for res in event:
            result.append(res.get())
    df = pd.DataFrame(result)
    counts = df.loc[:,:'comp'].groupby(['e','b']).sum() # sum counts
    counts.to_csv(out_file, mode = 'a', 
                  header = os.path.isfile(out_file) == False or os.path.getsize(out_file) == 0)
    return


def trace(a):
    '''
    Make 3-d trace of the trajectory
    '''
    # Coordinate transformation
    m1,m2,m3 = a.m1, a.m2, a.m3
    r = a.r

    mtot = m1 + m2 + m3
    c1 = m1/(m1+m2)
    c2 = m2/(m1+m2)
    r1 = np.array([-c2*r[i] - m3/mtot*r[i+3] for i in range(0,3)]) #x,y,z for particle 1
    r2 = np.array([c1*r[i]-m3/mtot*r[i+3] for i in range(0,3)])
    r3 = np.array([(m1+m2)/mtot*r[i+3] for i in range(0,3)])


    ax = plt.axes(projection='3d')
    ax.plot3D(r1[0], r1[1], r1[2], 'g')
    ax.plot3D(r2[0], r2[1], r2[2], 'orange')
    ax.plot3D(r3[0], r3[1], r3[2], 'r')
    ax.scatter3D(r1[0][0], r1[1][0], r1[2][0],marker = 'o',color = 'g')
    ax.scatter3D(r2[0][0], r2[1][0], r2[2][0],marker = 'o',color = 'orange')
    ax.scatter3D(r3[0][0], r3[1][0], r3[2][0],marker = 'o',color = 'r')
    ax.scatter3D(r1[0][-1], r1[1][-1], r1[2][-1],marker = '^',color = 'g')
    ax.scatter3D(r2[0][-1], r2[1][-1], r2[2][-1],marker = '^',color = 'orange')
    ax.scatter3D(r3[0][-1], r3[1][-1], r3[2][-1],marker = '^',color = 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return 