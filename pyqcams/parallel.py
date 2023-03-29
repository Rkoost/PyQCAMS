import pymar
import numpy as np
import utils
import os

# def save_long(n_traj,cpus,calc,out_file):
#     '''
#     Saves a long version of the data; one line per trajectory.
#     Stores all input & output data.
    
#     n_traj, int
#         number of trajectories
#     cpus, int
#         number of cpus for parallel calculation
#     calc, dict
#         output from pymar.start() function
#     out_file, string
#         output file name
#     '''
#     with mp.Pool(processes = cpus) as p:
#         event = [p.apply_async(pymar.main, kwds = (calc)) for i in range(n_traj)]
#         for res in event:
#             df = pd.DataFrame([res.get()])
#             df.to_csv(out_file, mode = 'a', index = False, 
#                     header = os.path.isfile(out_file) == False or os.path.getsize(out_file) == 0)
#     return

# def save_short(n_traj,cpus,calc, out_file):
#     '''
#     Saves a short version of the data; sums up the counts so one line per (E,b) set.
#     Use this if input data & distributions are not needed.

#     n_traj, int
#         number of trajectories
#     cpus, int
#         number of cpus for parallel calculation
#     calc, dict
#         output from pymar.start() function
#     out_file, string
#         output file name
#     '''
#     result = []
#     with mp.Pool(processes = cpus) as p:
#         event = [p.apply_async(pymar.main, kwds = (calc)) for i in range(n_traj)]
#         for res in event:
#             result.append(res.get())
#     df = pd.DataFrame(result)
#     clist = ['e','b','h2','cah1','cah2','diss']
#     counts = df.loc[:,clist].groupby(['e','b']).sum() # sum counts
#     counts.to_csv(out_file, mode = 'a',
#                   header = os.path.isfile(out_file) == False or os.path.getsize(out_file) == 0)
#     return

if __name__ == '__main__':
    calc = pymar.start('inputs.json') # Calculated parameters for main function

    n_traj = 10 # Number of trajectories
    out_file = f'example/results_short.csv'
    cpus = os.cpu_count() # Number of cpus for parallel calculation
    bvals = np.arange(0,1,.25) # Range of impact parameters

    # loop over all impact parameters
    for b in bvals:
        calc['b'] = b # set new impact parameter
        # utils.save_long(n_traj, cpus, calc, f'{out_file}') # Uncomment for long output
        utils.save_short(n_traj, cpus, calc, f'{out_file}') # Uncomment for short output
    