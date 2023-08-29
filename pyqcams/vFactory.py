import numpy as np
from scipy.optimize import fsolve

def morse(de = 1.,alpha = 1.,re = 1.):
    '''Usage:
            V = morse(**kwargs)
    
    Return a one-dimensional morse potential:
    V(r) = De*(1-exp(-a*(r-re)))^2 - De

    Keyword arguments:
    de, float
        dissociation energy (depth)
    alpha, float
        returned from eVib function
    re, float
        equilibrium length
    '''
    V = lambda r: de*(1-np.exp(-alpha*(r-re)))**2 - de
    dV = lambda r: 2*alpha*de*(1-np.exp(-alpha*(r-re)))*np.exp(-alpha*(r-re))
    return V, dV

def lj(m=12, n = 6, cm = 1., cn=1., **kwargs):
    '''Usage:
        V = lj(**kwargs)
    
    Return a one-dimensional general Lennard-Jones potential:
    V(r) = cm/r^m - cn/r^n

    Keyword arguments:
    cn, float
        long-range parameter
    cm, float
        short-range parameter
    '''
    V = lambda r: cm/r**(m)-cn/r**(n)
    dV = lambda r: -m*cm/r**(m+1)+n*cn/r**(n+1)
    return V, dV

def buckingham(a=1., b=1., c6 = 1., max = .1,**kwargs):
    '''Usage:
        V = buckingham(**kwargs)
    Buckingham potentials tend to come back down at low r. 
    We fix this by imposing xmin at the turning point "max."
    Return a one-dimensional Buckingham potential:
    V(r) = a*exp(-b*r) - c6/r^6 for r > r_x

    Keyword arguments:
    a, float
        short-range multiplicative factor
    b, float
        short-range exponential factor
    c6, float
        dispersion coefficient
    tp, float
        guess of turning point
    xmin, float
        minimum of potential (cutoff at local maximum)        
    xmax, float
        maximum of potential

    Outputs:
    Buck, function
        buckingham potential
    dBuck, function
        derivative of buckingham potential
    xi, float
        minimum x-value where Buck is defined
    '''
    Buck = lambda r: a*np.exp(-b*r) - c6/r**6
    dBuck = lambda r: -a*b*np.exp(-b*r) + 6*c6/r**7
    ddBuck = lambda r: a*b**2*np.exp(-b*r) - 6*7*c6/r**8

    # Find the maximum of potential
    xi = fsolve(dBuck,max)
    
    return Buck, dBuck, xi

def axilrod(C = 0):
    '''
    Return Axilrod-Teller potential
    
    C = V*alpha1*alpha2*alpha3

    V - Ionization energy 
    alpha - atomic polarizability


    '''
    V = lambda r12,r23,r31: C*(1/(r12*r23*r31)**3 - 3*((r12**2-r23**2-r31**2)*
                                                        (r23**2-r31**2-r12**2)*
                                                        (r31**2-r12**2-r23**2))/
                                                    8/(r12*r23*r31)**5)
    
    # Partial derivative w.r.t. x (symmetry between r12, r23, r31)
    # dV = lambda x,y,z: -C*(3*((x**6 + x**4*(y**2 + z**2) - 
    #                                     5*(y**2 - z**2)**2*(y**2 + z**2) + 
    #                                     x**2*(3*y**4 + 2*y**2*z**2 + 3*z**4)))/
    #                                 (8*x**6*y**5*z**5))
    
    dV = lambda x,y,z: -3*C*(x**6 + x**4*(y**2 + z**2) - 
                        5*(y**2 - z**2)**2*(y**2 + z**2) + 
                        x**2*(3*y**4 + 2*y**2*z**2 + 3*z**4))/(8*x**6*y**5*z**5)
    return V, dV

