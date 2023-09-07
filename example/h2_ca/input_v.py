import numpy as np
from scipy.optimize import fsolve
import sys
import pyqcams.vFactory as vF


# sys.path.insert(0, '.')
# import pyqcams.vFactory as vF

# Samples of the vFactory potentials for CaH2 are given, uncomment for use.

# Diatomic potentials and their derivatives.

# v12, dv12= vF.morse(de = 0.16456603489, re = 1.40104284795, alpha = 1.059493476908482)  # H2
# v23, dv23= vF.morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358) # CaH
# v31, dv31= vF.morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358) # CaH

# v12, dv12= vF.lj(cm = 64.16474114146757, cn = 6.49902670540583931313)  # H2
# v23, dv23= vF.lj(cm = 38365.980245558436, cn = 100.1) # CaH
# v31, dv31= vF.lj(cm = 38365.980245558436, cn = 100.1) # CaH

# v12, dv12, xmin12 = vF.buckingham(a = 167205.03207304262,b= 8.494089813101883, 
#                          c6= 6.49902670540583931313, re= 1.6,max= 0.2)  # H2
# v23, dv23, xmin23 = vF.buckingham(a= 508.162571320063,b =2.820905669626857,
#                          c6=100.1, max= 2) # CaH
# v31, dv31, xmin31 = vF.buckingham(a= 508.162571320063,b =2.820905669626857,
#                          c6=100.1, max= 2) # CaH

# Polyatomic fit to ab initio data (10.1063/1.462163)
c12 = [(-6.48477958e-01,1), (6.20604981e-01,2),(-2.42153657e-01,3),(4.87242893e-02, 4),
       (-5.17469520e-03,5), (2.44721392e-04,6), (-2.80778902e-07,8)]
c23 = [ (-1.60984437e+01,1), (3.48288096e+01,2), (-3.17593909e+01,3),  (1.55108703e+01,4), 
       (-4.19952982e+00,5),(5.41459871e-01,6), (-6.62095667e-03,8), (9.23374170e-05,10), (-6.83472726e-07,12)]
c31 = [ (-1.60984437e+01,1), (3.48288096e+01,2), (-3.17593909e+01,3),  (1.55108703e+01,4), 
       (-4.19952982e+00,5),(5.41459871e-01,6), (-6.62095667e-03,8), (9.23374170e-05,10), (-6.83472726e-07,12)]
v12, dv12 = vF.poly2(1.07964274e+00, 2.34532220e+00, 4.36857937e-02, c12)
v23, dv23 = vF.poly2(2.01524188e+01,  1.60763936e+00,  7.91913435e-02, c23)
v31, dv31 = vF.poly2(2.01524188e+01,  1.60763936e+00,  7.91913435e-02, c31)


# Range of potentials
# If using Buckingham potential, set xmin at returned value from vF.buckingham()
r12 = np.linspace(0.5, 20, 1000) # min, max, number of points
r23 = np.linspace(2, 20, 1000)
r31 = np.linspace(2, 20, 1000)

# Equilibrium point solutions are required for each diatomic potential.
# Use best guess as 2nd argument of fsolve.
req_12 = fsolve(dv12, 1.0) # H2
req_23 = fsolve(dv23, 3.0) # CaH
req_31 = fsolve(dv31, 3.0) # CaH

# Three-body PES. 

# Axilrod-Teller potential.
# C = 10 
# v123, dv123dR12, dv123dR23, dv123dR31 = vF.axilrod(C)

# Polyatomic fit to ab initio data (10.1063/1.462163)

# Sample of a tentative 3 - body fitting function
pow = ['111','110','011','120','021', 
       '201','121','211','220','022', 
       '130','031','301','221','122', 
       '131','311','230','032','302']
d =    [ 0.07317053,  0.1589012 , -0.05797962,
       -0.2932295 ,  0.21494367, -0.02749947,  0.0301055 , -0.01672027,
        0.02517304, -0.04913106,  0.06304716, -0.05665921,  0.00181586,
        0.00548205, -0.00252834, -0.0096243 , -0.00168267, -0.00582695,
        0.01406624,  0.00116524]
coeff = list(zip(d,[[int(j) for j in i] for i in pow]))

v123, dv123dr12, dv123dr23, dv123dr31 = vF.poly3(0.07756418,  0.09746985, 0.09746985, coeff)

if __name__ == '__main__':
    # Take a look at the potentials, make sure we have a good guess for re. 
    import matplotlib.pyplot as plt
    plt.plot(r12, v12(r12), label = r'$V_{12}$') # H2 potential
    plt.plot(r23, v23(r23), label = r'$V_{23}$') # CaH potential
    plt.plot(r31, v31(r31), label = r'$V_{31}$')
    plt.plot(req_12, v12(req_12), 'o')
    plt.plot(req_23, v23(req_23), 'o')
    plt.plot(req_31, v31(req_31), 'o')
    plt.legend()
    plt.xlabel(r'$R (a_0)$')
    plt.ylabel(r'V ($E_H$)')
    plt.show()

    N = 100
    rab = np.linspace(r12.min(), r12.max(), N)
    rbc = np.linspace(r23.min(), r23.max(), N)    
    rca = np.linspace(r31.min(), r31.max(), N)    
    X, Y = np.meshgrid(rab, rbc)
    Z = v123(X,Y,0.5)   # Keep r31 constant
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.plot_surface(X, Y, Z)    # Plot 3d surface
    ax.set_xlabel(r'$r_{12}$')
    ax.set_ylabel(r'$r_{23}$')
    ax.set_zlabel(r'V ($E_H$)')
    plt.show()