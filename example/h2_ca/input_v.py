import numpy as np
from scipy.optimize import fsolve
import pyqcams.vFactory as vF


# Diatomic potentials and their derivatives.
# Samples of the 3 analytic potentials are given, uncomment for use.

v12, dv12= vF.morse(de = 0.16456603489, re = 1.40104284795, alpha = 1.059493476908482)  # H2
v23, dv23= vF.morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358) # CaH
v31, dv31= vF.morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358) # CaH

# v12, dv12= vF.lj(cm = 64.16474114146757, cn = 6.49902670540583931313)  # H2
# v23, dv23= vF.lj(cm = 38365.980245558436, cn = 100.1) # CaH
# v31, dv31= vF.lj(cm = 38365.980245558436, cn = 100.1) # CaH

# v12, dv12, xmin12 = vF.buckingham(a = 167205.03207304262,b= 8.494089813101883, 
#                          c6= 6.49902670540583931313, re= 1.6,max= 0.2)  # H2
# v23, dv23, xmin23 = vF.buckingham(a= 508.162571320063,b =2.820905669626857,
#                          c6=100.1, re= 3.1, max= 2) # CaH
# v31, dv31, xmin31 = vF.buckingham(a= 508.162571320063,b =2.820905669626857,
#                          c6=100.1, re= 3.1, max= 2) # CaH


# Range of potentials
# NOTE: if using Buckingham potential, set xmin at left turning point
r12 = np.linspace(0.5, 20, 1000) # min, max, number of points
r23 = np.linspace(2, 20, 1000)
r31 = np.linspace(2, 20, 1000)

# Equilibrium point solutions are required for each diatomic potential.
# Use best guess as 2nd argument of fsolve.
req_12 = fsolve(dv12, 1.6) # H2
req_23 = fsolve(dv23, 2) # CaH
req_31 = fsolve(dv31, 2) # CaH

# Three-body PES. This example uses the Axilrod-Teller potential, 
# which is available in vFactory, but written as a lambda function here. 
C = 10 # 0 to turn of 3-body interaction

def vTrip(r12, r23, r31):
    V = C*(1/(r12*r23*r31)**3 - 3*((r12**2-r23**2-r31**2)*
                                (r23**2-r31**2-r12**2)*
                                (r31**2-r12**2-r23**2))/
                                8/(r12*r23*r31)**5) 
    return V

# vTrip = lambda r12,r23,r31: C*(1/(r12*r23*r31)**3 - 3*((r12**2-r23**2-r31**2)*
#                                                        (r23**2-r31**2-r12**2)*
#                                                        (r31**2-r12**2-r23**2))/
#                                                         8/(r12*r23*r31)**5) 

dv123dR12 = lambda r12,r23,r31: -3*C*(r12**6 + r12**4*(r23**2 + r31**2) - 
            5*(r23**2 - r31**2)**2*(r23**2 + r31**2) + 
            r12**2*(3*r23**4 + 2*r23**2*r31**2 + 3*r31**4))/(8*r12**6*r23**5*r31**5)

dv123dR23= lambda r12,r23,r31: -3*C*(r23**6 + r23**4*(r31**2 + r12**2) - 
            5*(r31**2 - r12**2)**2*(r31**2 + r12**2) + 
            r23**2*(3*r31**4 + 2*r31**2*r12**2 + 3*r12**4))/(8*r23**6*r31**5*r12**5)

dv123dR31= lambda r12,r23,r31: -3*C*(r31**6 + r31**4*(r12**2 + r23**2) - 
            5*(r12**2 - r23**2)**2*(r12**2 + r23**2) + 
            r31**2*(3*r12**4 + 2*r12**2*r23**2 + 3*r23**4))/(8*r31**6*r12**5*r23**5)

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

