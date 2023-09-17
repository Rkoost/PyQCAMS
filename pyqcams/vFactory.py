import numpy as np
from scipy.optimize import fsolve

# Two body potentials
def morse(de = 1.,alpha = 1.,re = 1.):
    '''Usage:
            V = morse(**kwargs)
    
    Return a one-dimensional morse potential:
    V(r) = De*(1-exp(-a*(r-re)))^2 - De

    Keyword arguments:
    de, float
        dissociation energy (depth)
    alpha, float
        alpha = we*sqrt(mu/2De)
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
    max, float
        guess of where the maximum is. At short range, 
        Buckingham potentials can reach a maximum and collapse. 
        Enter your nearest $r$ value to this maximum.

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

def poly2(c0, alpha, b, coeff):
    '''
    Polynomial fit of 2-body ab-initio data (10.1063/1.462163)
    V(r) = c0*e^(-alpha*x)/x + sum(c_i*rho^i), rho = x*e^-(b*x)
    
    Inputs:
    c0, float
    alpha, float
    b, float
    coeff, zipped list of the format [(c_i,i)] 
        Keep track of coeffiecients (c_i) and degree (i)
    '''
    v_long = lambda x: sum([i*((x*np.exp(-b*x))**j) for i, j in coeff])
    v_short = lambda x: c0*np.exp(-alpha*x)/x 
    v = lambda x: v_long(x) + v_short(x)
    dv_long = lambda x: sum([i*(-j*(b*x-1)*x**(j-1)*np.exp(-j*b*x)) for i, j in coeff])
    dv_short = lambda x: -c0*(1+alpha*x)*(np.exp(-alpha*x)/x**2)
    dv = lambda x: dv_long(x) + dv_short(x)
    return v, dv

# Three body potentials
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
    
    dvdR12 = lambda r12,r23,r31: -3*C*(r12**6 + r12**4*(r23**2 + r31**2) - 
            5*(r23**2 - r31**2)**2*(r23**2 + r31**2) + 
            r12**2*(3*r23**4 + 2*r23**2*r31**2 + 3*r31**4))/(8*r12**6*r23**5*r31**5)
    
    dvdR23= lambda r12,r23,r31: -3*C*(r23**6 + r23**4*(r31**2 + r12**2) - 
            5*(r31**2 - r12**2)**2*(r31**2 + r12**2) + 
            r23**2*(3*r31**4 + 2*r31**2*r12**2 + 3*r12**4))/(8*r23**6*r31**5*r12**5)
    
    dvdR31= lambda r12,r23,r31: -3*C*(r31**6 + r31**4*(r12**2 + r23**2) - 
            5*(r12**2 - r23**2)**2*(r12**2 + r23**2) + 
            r31**2*(3*r12**4 + 2*r12**2*r23**2 + 3*r23**4))/(8*r31**6*r12**5*r23**5)
    return V, dvdR12, dvdR23, dvdR31


def poly3(b_12, b_23, b_31, coeffs):
    '''
    Polynomial fit of 3-body ab-initio data (10.1063/1.462163)
    sum_{ijk}^{M} (d_ijk*p_12^i*p_23^j*p_31^k), p_ab = x_ab*e^(-b*x_ab)

    ENSURE i+j+k!=i!=j!= k AND i+j+k<=M
    Inputs:
    b12, b23, b31, float
        exponential parameters of p_ab
    coeffs, zipped list of the format [(d_ijk,[i,j,k])] 
        Keep track of coeffiecients and degree
    '''
    v = lambda r12,r23,r31: sum([i*(r12*np.exp(-b_12*r12))**j[0]*(r23*np.exp(-b_23*r23))**
                                 j[1]*(r31*np.exp(-b_31*r31))**j[2] for i,j in coeffs])
    dvdr12 = lambda r12,r23,r31: sum([-j[0]*(b_12*r12-1)/r12*i*(r12*np.exp(-b_12*r12))**
                                      j[0]*(r23*np.exp(-b_23*r23))**
                                      j[1]*(r31*np.exp(-b_31*r31))**j[2] for i, j in coeffs])
    dvdr23 = lambda r12,r23,r31: sum([-j[1]*(b_23*r23-1)/r23*i*(r12*np.exp(-b_12*r12))**
                                      j[0]*(r23*np.exp(-b_23*r23))**
                                      j[1]*(r31*np.exp(-b_31*r31))**j[2] for i, j in coeffs])
    dvdr31 = lambda r12,r23,r31: sum([-j[2]*(b_31*r31-1)/r31*i*(r12*np.exp(-b_12*r12))**
                                      j[0]*(r23*np.exp(-b_23*r23))**
                                      j[1]*(r31*np.exp(-b_31*r31))**j[2] for i, j in coeffs])
    return v, dvdr12, dvdr23, dvdr31
