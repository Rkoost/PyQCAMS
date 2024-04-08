import numpy as np
import warnings
from scipy.optimize import root_scalar
from pyqcams.qct import *
from pyqcams.constants import *


def jac2cart(x, C1, C2):
    '''
    Jacobian to Cartesian coordinate system. 
    x contains all Jacobi position vectors,
    C1 = m1/(m1+m2)
    C2 = m2/(m1+m2)
    '''
    rho1x, rho1y, rho1z, rho2x, rho2y, rho2z = x
    
    # Internuclear distances 
    r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)
    r23 = np.sqrt((rho2x - C1*rho1x)**2
                + (rho2y - C1*rho1y)**2 
                + (rho2z - C1*rho1z)**2)
    r31 = np.sqrt((rho2x + C2*rho1x)**2
                + (rho2y + C2*rho1y)**2 
                + (rho2z + C2*rho1z)**2)
    return r12, r23, r31    

def hamiltonian(traj):
        '''Hamiltonian; returns energy and angular momentum as a function of
            internuclear distances and momenta.
        Use:
            Check for conservation of energy/momentum.

        vi, function
            potential describing initial molecule (H2)
        vf, function
            potential describing reaction-formed molecule (CaH)
        w, list
            state variables; w = [rho1x, rho1y, rho1z, rho2x, rho2y, rho2z,
                                p1x, p1y, p1z, p2x, p2y, p2z]
        p, list
            state parameters; p = [traj.m1, traj.m2, traj.m3, traj.mu12, traj.mu23, traj.mu31]
        Math:
        E = KE + V = p^2/2m + V

        Returns:
        etot, float
            total energy
        ll, float
            total angular momentum
        ekin, float
            kinetic energy
        epot, foat
            potential energy
        '''
        w = traj.wn

        rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, \
            p1x, p1y, p1z, p2x, p2y, p2z = w
        
        r12,r23,r31 = jac2cart(w[:6],traj.C1,traj.C2)
        
        # Kinetic energy
        ekin = 0.5*(p1x**2+p1y**2+p1z**2)/traj.mu12 \
                + 0.5*(p2x**2+p2y**2+p2z**2)/traj.mu312

        # Potential energy
        epot = np.asarray(traj.v1(r12))+np.asarray(traj.v2(r23)) \
             + np.asarray(traj.v3(r31)) + np.asarray(traj.vtrip(r12,r23,r31))

        # Total energy
        etot = ekin + epot

        # Radial angular momenta
        # Lx = Lx1 + Lx2

        lx1 = rho1y*p1z - rho1z*p1y # Internal angular momentum
        lx2 = rho2y*p2z - rho2z*p2y # Relative angular momentum
        lx = lx1 + lx2 

        ly1 = rho1z*p1x - rho1x*p1z
        ly2 = rho2z*p2x - rho2x*p2z
        ly = ly1 + ly2 

        lz1 = rho1x*p1y - rho1y*p1x
        lz2 = rho2x*p2y - rho2y*p2x
        lz = lz1 + lz2 

        ll = np.sqrt(lx**2 + ly**2 + lz**2)

        return (etot, epot, ekin, ll)

def get_results(traj, *args):
    results =  {'e': traj.E0/K2Har,
                'b': traj.b,
                'vi': traj.vi,
                'ji': traj.ji,
                'n12': traj.count[0],
                'n23': traj.count[1],
                'n31': traj.count[2],
                'nd': traj.count[3],
                'nc': traj.count[4],
                'v': traj.fstate[0],
                'vw': f'{traj.fstate[1]:.3e}',
                'j': traj.fstate[2],
                'jw': f'{traj.fstate[3]:.3e}'}
    for arg in args:
         results[arg] = getattr(traj,arg)
    return results