import numpy as np
# import pyqcams.util as utils
import os
from pyqcams import pymar, psave

if __name__ == '__main__':
    calc = pymar.start('example/h2_ca/') # Calculated parameters for main function

    n_traj = 100 # Number of trajectories
    out_file = f'example/h2_ca/results_short.csv'
    cpus = os.cpu_count() # Number of cpus for parallel calculation
    bvals = np.arange(0,1,.25) # Range of impact parameters

    # loop over all impact parameters
    for b in bvals:
        calc['b'] = b # set new impact parameter
        #psave.save_long(n_traj, cpus, calc, f'{out_file}') # Uncomment for long output
        psave.save_short(n_traj, cpus, calc, f'{out_file}') # Uncomment for short output
    
