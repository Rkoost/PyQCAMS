import numpy as np
from pyqcams import qct, constants, potentials

# Masses
m1 = 1.008*constants.u2me
m2 = 1.008*constants.u2me
m3 = 40.078*constants.u2me

# Collision parameters
E0 = 20000 # collision energy (K)
b0 = 0
R0 = 50 # Bohr

# Potential parameters in atomic units
v12, dv12 = potentials.morse(de = 0.16456603489, re = 1.40104284795, alpha = 1.059493476908482)
v23, dv23 = potentials.morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358)
v31, dv31 = potentials.morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358)

# Three-body interaction
v123, dv123dr12, dv123dr23, dv123dr31 = potentials.axilrod(C=0)

# Initiate molecules
mol12 = qct.Molecule(mi = m1, mj = m2, Vij = v12, dVij = dv12, xmin = .5, xmax = 30,
                     vi = 1, ji = 0, npts=1000)
mol23 = qct.Molecule(mi = m2, mj = m3, Vij = v23, dVij = dv23, xmin = 1, xmax = 40)
mol31 = qct.Molecule(mi = m3, mj = m1, Vij = v31, dVij = dv31, xmin = 1, xmax = 40)

input_dict = {'m1':m1,'m2':m2,'m3':m3,
    'E0': E0, 'b0': b0, 'R0': R0, 'seed': None,
    'mol_12': mol12,'mol_23': mol23,'mol_31': mol31,
    'vt': v123, 'dvtdr12': dv123dr12, 'dvtdr23': dv123dr23, 'dvtdr31': dv123dr31,
    'integ':{'t_stop': 2, 'r_stop': 2, 'r_tol': 1e-10, 'a_tol':1e-8,'econs':1e-5,'lcons':1e-5}}


if __name__ == '__main__':
    # Run over a range of impact parameters.
    nTraj = 10
    bs = np.arange(0,5,0.25)
    for b in bs:
        input_dict['b0'] = b
        qct.runN(nTraj,input_dict,short_out='results/sampleshort.txt',long_out = 'results/samplelong.txt')
