'''
Methods to run parallel trajectories and save short or long.
'''


import os
import multiprocess as mp
import pandas as pd
from pyqcams import main

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
        output from main.start() function
    out_file, string
        output file name
    '''
    with mp.Pool(processes = cpus) as p:
        event = [p.apply_async(main.main, kwds = (calc)) for i in range(n_traj)]
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
        output from main.start() function
    out_file, string
        output file name
    '''
    result = []
    with mp.Pool(processes = cpus) as p:
        event = [p.apply_async(main.main, kwds = (calc)) for i in range(n_traj)]
        for res in event:
            result.append(res.get())
    df = pd.DataFrame(result)
    counts = df.loc[:,:'comp'].groupby(['e','b']).sum() # sum counts
    counts.to_csv(out_file, mode = 'a', 
                  header = os.path.isfile(out_file) == False or os.path.getsize(out_file) == 0)
    return

if __name__ == '__main__':
    calc = main.start('example/h2_ca/')
    