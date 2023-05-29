import numpy as np
# import pyqcams.util as utils
import os
from pyqcams import pymar, psave

if __name__ == '__main__':
    print(os.getcwd())
    calc = pymar.start('inputs.json') # Calculated parameters for main function

    n_traj = 10 # Number of trajectories
    out_file = f'example/results_short.csv'
    cpus = os.cpu_count() # Number of cpus for parallel calculation
    bvals = np.arange(0,1,.25) # Range of impact parameters

    # loop over all impact parameters
    for b in bvals:
        calc['b'] = b # set new impact parameter
        # utils.save_long(n_traj, cpus, calc, f'{out_file}') # Uncomment for long output
        psave.save_short(n_traj, cpus, calc, f'{out_file}') # Uncomment for short output
    