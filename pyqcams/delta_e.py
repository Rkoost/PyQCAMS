import multiprocess as mp
import pymar
import pandas as pd
import numpy as np
import plotters
import os

calc = pymar.start('inputs.json')
traj = pymar.QCT(**calc)
traj.runT()
# plotters.traj_3d(traj)
plotters.traj_gif(traj)