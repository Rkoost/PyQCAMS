import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import os
import tabula 
import pandas as pd
import json

def numToV(file):
    '''
    Convert pdf table to potential function and first derivative.
    Assumes positions in column 0

    Inputs:

    file, str
        path to pdf file
    pgs, list
        integer list of pages to convert
    col, int
        column containing energies
    '''
    types = ['.csv','.txt','.dat']
    split = os.path.splitext(f'{file}')
    filetype = split[1]
    if filetype in types:
        try:
            df = pd.read_csv(f'{file}',header = None)
            num_x = np.array([float(i) for i in df[0][:].values.tolist()])
            num_y = np.array([float(i) for i in df[1][:].values.tolist()])
        except:
            try:
                df = pd.read_csv(f'{file}',header = None, sep = '\s+')
                num_x = np.array([float(i) for i in df[0][:].values.tolist()])
                num_y = np.array([float(i) for i in df[1][:].values.tolist()])
            except:
                print('Please use either comma or tab separated data.')
    else: 
        print('Please specify filetype. Choose from "csv", "dat", or "txt".')
    num_V = InterpolatedUnivariateSpline(num_x,num_y, k = 4) # Use k = 4 to find roots of derivative
    num_dV = num_V.derivative()
    cr_pts = num_dV.roots()
    cr_pts = np.append(cr_pts, (num_x[0], num_x[-1]))  # also check the endpoints of the interval
    cr_vals = num_V(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals[min_index:])
    print(min_index)
    num_re = cr_pts[min_index] # Equilibrium distance
    num_y -= cr_vals[max_index] # Shift values to make curve approach 0
    num_V = InterpolatedUnivariateSpline(num_x,num_y, k = 4)
    return num_V, num_dV, num_re

if __name__ == '__main__':
    input_file = "example\\h2_ca\\inputs.json"
    with open(f'{input_file}') as f:
        data = json.load(f)
    potential_AB = data['potential_AB'] # which potential
    print(potential_AB)
    V, dV, re = numToV(f'{potential_AB}') # Find V, dV via spline
    xmin = data['potential_params']["AB"]['xmin'] 
    xmax = data['potential_params']["AB"]['xmax']
    # V, dV = numToV(r'C:\Users\Rian\Documents\research\jesusgroup\cah_pec\cah.txt')
    # V, dV = numToV(r'C:\Users\Rian\Documents\research\jesusgroup\cah_pec\h2_pec.dat')
    x = np.linspace(xmin,xmax,1000)
    plt.plot(x,(V(x)))
    plt.plot(re, V(re), marker = 'o')
    plt.plot(x, dV(x))
    plt.show()