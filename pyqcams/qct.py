import numpy as np
import scipy.linalg as linalg
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root_scalar, fsolve
import pandas as pd
import os, time
import matplotlib.pyplot as plt
from pyqcams import util, constants, analysis
import warnings
# from joblib import Parallel, delayed
import multiprocess as mp

class Molecule:
    '''
    The molecule object represents molecules.
    It is described by reduced mass, interaction potential, and internal state.
    This can represent an initial or final molecule.
    '''
    def __init__(self,**kwargs):
        self.mi = kwargs.get('mi')
        self.mj = kwargs.get('mj')
        self.mu = self.mi*self.mj/(self.mi + self.mj)
        self.vi = kwargs.get('vi') # Initial molecules
        self.ji = kwargs.get('ji')
        self.Ei = kwargs.get('Ei')
        self.Vij = kwargs.get('Vij')
        self.dVij = kwargs.get('dVij')
        self.xmin = kwargs.get('xmin')
        self.xmax = kwargs.get('xmax')
        self.npts = kwargs.get('npts')
        self.Veff = lambda x: self.Vij(x) + self.ji*(self.ji+1)/(2*self.mu*x**2)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'mu:{self.mu!r}, vi:{self.vi!r}, ji:{self.ji!r})')
    
    def get_vi(self):
        return self.vi

    def set_E(self,value):
        self.E = value

    def set_rp(self,value):
        self.rp = value

    def set_rm(self,value):
        self.rm = value

    def set_re(self,value):
        self.re = value
    
    def set_vPrime(self,value):
        self.vPrime = value

    def set_jPrime(self,value):
        self.jPrime = value
        # New Veff for new jprime
        self.Veff = lambda x: self.Vij(x) + self.jPrime*(self.jPrime+1)/(2*self.mu*x**2)

    def checkBound(self, rf):
        '''
        Check if the molecule is bound. Bound molecules 
        have a defined equilibrium point and have energy
        less than the boundary. 
        '''
        # Bound molecule has defined equilibrium
        if self.re is not None:
            if self.bdry == 0: # No rotation
                if self.E < 0: 
                    return True
                else:
                    return False
            elif (self.E < self.bdry) and (rf < self.bdx): # Rotation
                return True
            else:
                return False
        else:
            return False
        
    def DVR(self):
        '''
        Returns the energy spectrum for a potential energy with a bound state.
        '''
        xl = float(self.xmax-self.xmin)
        dx = xl/self.npts
        n = np.arange(1,self.npts)

        x = float(self.xmin) + dx*n
        VX = np.diag(self.Vij(x)) # Potential energy matrix
        _i = n[:,np.newaxis]
        _j = n[np.newaxis,:]
        m = self.npts + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore divide by 0 warn
            # Off-diagonal elements of kinetic energy matrix
            T = ((-1.)**(_i-_j)
                * (1/np.square(np.sin(np.pi/(2*m)*(_i-_j)))
                - 1/np.square(np.sin(np.pi/(2*m)*(_i+_j)))))
        # Diagonal elements of KE matrix
        T[n-1,n-1] = 0 
        T += np.diag((2*m**2+1)/3
             -1/np.square(np.sin(np.pi*n/m)))
        T *= np.pi**2/4/xl**2/self.mu
        HX = T + VX # Hamiltonian 
        # evals, evecs = np.linalg.eigh(HX) # Solve the eigenvalue problem
        evals, evecs = linalg.eigh(HX)
        # evals = linalg.eigvalsh(HX)
        # if self.j == 0:
        #     evals = linalg.eigvalsh(HX, subset_by_index=[0,self.neigs])
        
        # To include the rotational coupling term 
        # E(v,j) = we*(v+.5) + wexe*(v+.5)^2 + bv*j*(j+1)
        # Where bv = be - ae*(v+.5) = (hbar^2/2m)<psi_v|1/r^2|psi_v>
        # We calculate the expectation value of the rotational energy of a 
        # vibrational eigenstate evecs
        # print((evecs[:,]**2).sum(axis=1)) # Check evecs is normalized
        # bv = []
        # for i in range(evecs.shape[1]):
        #     bv.append(np.trapz(evecs[:,i]**2/(x**2),dx=dx)/2/self.mu/dx)
        # bv = np.asarray(bv)
        # bv *= self.j*(self.j+1)
        self.evj = evals #+ bv
        return evals, evecs

    def rebound(self):
        '''
        Find re, bound point for j > 0
        '''
        if (hasattr(self,'jPrime')) and (self.jPrime is not None):
            dVeff = lambda x: self.dVij(x) - self.jPrime*(self.jPrime+1)/(self.mu*x**3)
        else:
            dVeff = lambda x: self.dVij(x) - self.ji*(self.ji+1)/(self.mu*x**3)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Ignore too many calls
                re = fsolve(dVeff, self.xmin)
                self.set_re(re) # set re
            # If no bound states exist, and large root is found
            if self.re > self.xmax:
                self.set_re(None)
            elif self.Veff(self.re) > 0:
                raise Exception('Equilibrium distance not found.')
        except Exception:
            self.set_re(None)
        # For j > 0
        if self.re is not None:
            try:
                self.bdx = root_scalar(dVeff, bracket = [self.re*1.001,self.xmax]).root
                self.bdry = self.Veff(self.bdx)
            except ValueError: # If j=0, boundary is 0
                self.bdry = 0

    def turningPts(self, initial = False):
        '''
        Set classical turning points and vprime for the product molecule.
        initial, Boolean
            True if initial molecule,
            False if product molecule

        '''
        self.rebound() # Find re, bound
        
        if initial:
            # If spectrum is not known, use DVR
            if self.Ei is None:
                self.DVR() # Set evj attribute
                self.Ei = self.evj[self.vi] # Set internal rovib energy

            diff = lambda x: self.Ei - self.Vij(x) - self.ji*(self.ji+1)/(2*self.mu*x**2)
            # Find outer turning point
            try:
                if self.bdry == 0:
                    self.rp = root_scalar(diff, bracket = [self.re, self.xmax]).root # set rplus
                else:
                    self.rp = root_scalar(diff, bracket = [self.re, self.bdx]).root # set rplus
            except:
                raise Exception(f'No outer turning point found, energy is too high.')

            # Find inner turning point
            self.rm = root_scalar(diff, bracket = [self.xmin, self.re]).root # set rminus
            
            # Set oscillation period, vprime
            self.tau = quad(lambda x: 1/np.sqrt(diff(x)),self.rm,self.rp)[0]
            self.tau *=np.sqrt(2*self.mu)
            vib = quad(lambda x: np.sqrt(diff(x)),self.rm,self.rp)[0]
            vib*=np.sqrt(2*self.mu)/np.pi
            vib+= -0.5 
            self.vi = np.round(vib)
            
        else:
            diff = lambda x: self.E - self.Vij(x) - self.jPrime*(self.jPrime+1)/(2*self.mu*x**2)
            # Find outer turning point
            # Sometimes E < Veff(re)
            if diff(self.re) < 0:
                raise Exception(f'E below minimum of potential.')
            try:
                if self.bdry == 0:
                    self.rp = root_scalar(diff, bracket = [self.re, self.xmax]).root # set rplus
                else:
                    self.rp = root_scalar(diff, bracket = [self.re, self.bdx]).root # set rplus
            except:
                raise Exception(f'No outer turning point found, energy is too high.')

            # Find inner turning point
            self.rm = root_scalar(diff, bracket = [self.xmin, self.re]).root # set rminus
            
            # Set vprime
            vib = quad(lambda x: np.sqrt(diff(x)),self.rm,self.rp)[0]
            vib*=np.sqrt(2*self.mu)/np.pi
            vib+= -0.5 
            self.set_vPrime(vib)
    
    def gaussBin(self,j_eff):
        '''
        Gaussian bin for the molecule's rovibrational number, 
        with sigma = 0.05
        Returns:
        w, float
            weight associated with vibrational product
        '''
        self.vt = np.round(self.vPrime)
        self.vw = np.exp(-np.abs(self.vPrime - self.vt)**2/0.05**2)
        self.vw *= 1/np.sqrt(np.pi)/0.05

        # j is rounded to the nearest integer
        self.jw = np.exp(-np.abs(j_eff - self.jPrime)**2/0.05**2)
        self.jw *= 1/np.sqrt(np.pi)/0.05


class Trajectory:
    def __init__(self,**kwargs):
        self.mol_12 = kwargs.get('mol_12') # Initial 12 molecule
        self.mol_23 = kwargs.get('mol_23') # 23 molecule
        self.mol_31 = kwargs.get('mol_31') # 31 molecule
        self.vi = self.mol_12.vi
        self.ji = self.mol_12.ji
        self.m1 = kwargs.get('m1')
        self.m2 = kwargs.get('m2')
        self.m3 = kwargs.get('m3')
        self.E0 = kwargs.get('E0')*constants.K2Har
        self.b = kwargs.get('b0')
        self.R0 = kwargs.get('R0')
        self.v1 = self.mol_12.Vij
        self.v2 = self.mol_23.Vij
        self.v3 = self.mol_31.Vij
        self.vtrip = kwargs.get('vt')
        self.dvtdr12 = kwargs.get('dvtdr12')
        self.dvtdr23 = kwargs.get('dvtdr23')
        self.dvtdr31 = kwargs.get('dvtdr31')
        self.seed = kwargs.get('seed')
        self.t_stop = kwargs.get('integ')['t_stop']
        self.r_stop = kwargs.get('integ')['r_stop']
        self.a_tol = kwargs.get('integ')['a_tol']
        self.r_tol = kwargs.get('integ')['r_tol']
        self.econs = kwargs.get('integ')['econs']
        self.lcons = kwargs.get('integ')['lcons']
        self.mtot = self.m1 + self.m2 + self.m3
        self.mu12 = self.m1*self.m2/(self.m1+self.m2) 
        self.mu23 = self.m2*self.m3/(self.m2+self.m3) 
        self.mu31 = self.m1*self.m3/(self.m1+self.m3) 
        self.mu312 = self.m3*(self.m1+self.m2)/self.mtot
        self.C1 = self.m1/(self.m1 + self.m2)
        self.C2 = self.m2/(self.m1 + self.m2)

    def set_attrs(self):
        self.delta_e = [np.nan]
        self.delta_l = [np.nan]
        self.wn = [np.nan]
        self.t = [np.nan]*2

    def set_fstate(self,value):
        # Set final state for bound molecule
        self.fstate = value

    def iCond(self):
        '''
        Generate initial conditions for the system.
        '''
        # Reset initial molecule jPrime
        self.mol_12.set_jPrime(self.ji)
        # Set turning points for mol_12
        self.mol_12.turningPts(initial = True)

        # Momentum for relative coordinate
        p0 = np.sqrt(2*self.mu312*self.E0)
        p1 = np.sqrt(self.mol_12.ji*(self.mol_12.ji+1))/self.mol_12.rp

        # Initial distance between atom and center of molecule
        # tau is vibrational period of the initial molecule
        rng = np.random.default_rng(self.seed)
        self.R = self.R0 + rng.random()*self.mol_12.tau*p0/self.mu312
        
        # Jacobi coordinates
        # Atom (relative coordinate)
        rho2x = 0
        rho2y = self.b # impact parameter
        rho2z = -np.sqrt(self.R**2-self.b**2)

        # Conjugate momenta for relative coords rho2
        p2x = 0
        p2y = 0
        p2z = p0 

        # Define collision parameters
        theta = np.arccos(1-2*rng.random())
        phi = 2*np.pi*rng.random()
        eta = 2*np.pi*rng.random()

        self.ang = [theta,phi,eta] # Store as input data

        # Molecule (internal coordinate)
        rho1x = self.mol_12.rp*np.sin(theta)*np.cos(phi)
        rho1y = self.mol_12.rp*np.sin(theta)*np.sin(phi)
        rho1z = self.mol_12.rp*np.cos(theta)

        # Conjugate momenta for internal coords rho1
        p1x = p1*(np.sin(phi)*np.cos(eta)
                - np.cos(theta)*np.cos(phi)*np.sin(eta))
        p1y = -p1*(np.cos(phi)*np.cos(eta)
                + np.cos(theta)*np.sin(phi)*np.sin(eta))
        p1z = p1*np.sin(theta)*np.sin(eta)

        self.w0 = np.array([rho1x,rho1y,rho1z,rho2x,rho2y,
                rho2z,p1x,p1y,p1z,p2x,p2y,p2z])

        return self.w0

    def hamEq(self,t,w):
        ''' Writes Hamilton's equations as a vector field. 
            Usage:
                Input function for scipy.integrate.solve_ivp
        t, None
            time
        w, list
            state variables; w = [rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, 
                                p1x, p1y, p1z, p2x, p2y, p2z]
        p, list
            state parameters; p = [m1, m2, m3, mu12, mu23, mu31, mu123]

        Math: qdot = dT/dp = d/dp(p^2/2mu) = p/mu; pdot = -dV/dq = -dV/dr*dr/dq

        Requires DERIVATIVES of potential functions. 
        '''
        rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
        r12, r23, r31 = util.jac2cart(w[:6], self.C1, self.C2)

        # Hamilton's equations (qdot)
        drho1x = p1x/self.mu12
        drho1y = p1y/self.mu12        
        drho1z = p1z/self.mu12 
        drho2x = p2x/self.mu312
        drho2y = p2y/self.mu312
        drho2z = p2z/self.mu312

        # Partial derivatives
        r12drho1x = rho1x/r12
        r12drho1y = rho1y/r12
        r12drho1z = rho1z/r12
        r23drho1x = (-self.C1*rho2x + self.C1**2*rho1x)/r23
        r23drho1y = (-self.C1*rho2y + self.C1**2*rho1y)/r23
        r23drho1z = (-self.C1*rho2z + self.C1**2*rho1z)/r23
        r23drho2x = (rho2x - self.C1*rho1x)/r23
        r23drho2y = (rho2y - self.C1*rho1y)/r23
        r23drho2z = (rho2z - self.C1*rho1z)/r23
        r31drho1x = (self.C2*rho2x + self.C2**2*rho1x)/r31
        r31drho1y = (self.C2*rho2y + self.C2**2*rho1y)/r31
        r31drho1z = (self.C2*rho2z + self.C2**2*rho1z)/r31
        r31drho2x = (rho2x+self.C2*rho1x)/r31
        r31drho2y = (rho2y+self.C2*rho1y)/r31
        r31drho2z = (rho2z+self.C2*rho1z)/r31

        dv12 = self.mol_12.dVij
        dv23 = self.mol_23.dVij
        dv31 = self.mol_31.dVij

        # Hamilton's equations (pdot)
        dP1x = - (dv12(r12)*r12drho1x + dv23(r23)*r23drho1x + dv31(r31)*r31drho1x
                + self.dvtdr12(r12,r23,r31)*r12drho1x + self.dvtdr23(r12,r23,r31)*r23drho1x + self.dvtdr31(r12,r23,r31)*r31drho1x)
        dP1y = - (dv12(r12)*r12drho1y + dv23(r23)*r23drho1y + dv31(r31)*r31drho1y
                + self.dvtdr12(r12,r23,r31)*r12drho1y + self.dvtdr23(r12,r23,r31)*r23drho1y + self.dvtdr31(r12,r23,r31)*r31drho1y)
        dP1z = - (dv12(r12)*r12drho1z + dv23(r23)*r23drho1z + dv31(r31)*r31drho1z
                + self.dvtdr12(r12,r23,r31)*r12drho1z + self.dvtdr23(r12,r23,r31)*r23drho1z + self.dvtdr31(r12,r23,r31)*r31drho1z)
        dP2x = - (dv23(r23)*r23drho2x + dv31(r31)*r31drho2x
                + self.dvtdr23(r12,r23,r31)*r23drho2x + self.dvtdr31(r12,r23,r31)*r31drho2x)
        dP2y = - (dv23(r23)*r23drho2y + dv31(r31)*r31drho2y
                + self.dvtdr23(r12,r23,r31)*r23drho2y + self.dvtdr31(r12,r23,r31)*r31drho2y)
        dP2z = - (dv23(r23)*r23drho2z + dv31(r31)*r31drho2z
                + self.dvtdr23(r12,r23,r31)*r23drho2z + self.dvtdr31(r12,r23,r31)*r31drho2z)
        
        # Hamilton's equations
        f = [drho1x, drho1y, drho1z, drho2x, drho2y, drho2z, 
             dP1x, dP1y, dP1z, dP2x, dP2y, dP2z]
        return f

    
    def runT(self):
        '''
        Run one trajectory.

        '''
        
        # Start fstate at 0
        self.set_fstate((0,0,0,0)) #v,vw,j,jw
        self.rejected = 0 # Keep track if trajectory fails
        self.count = [0,0,0,0,0] # n12,n23,n31,nd,nc
        self.iCond()
        self.vi = self.mol_12.get_vi()


        def stop1(t,w):
            '''Stop integration when r12 > "far" AU'''
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)

            return r12 - self.R*self.r_stop

        def stop2(t,w):
            '''Stop integration when r32 > "far" AU'''
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            r23 = np.sqrt((rho2x - self.C1*rho1x)**2
                        + (rho2y - self.C1*rho1y)**2 
                        + (rho2z - self.C1*rho1z)**2)
            return r23 - self.R*self.r_stop
        
        def stop3(t,w):
            '''Stop integration when r31 > "far" AU'''
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w 
            r31 = np.sqrt((rho2x + self.C2*rho1x)**2
                        + (rho2y + self.C2*rho1y)**2 
                        + (rho2z + self.C2*rho1z)**2)
            return r31 - self.R*self.r_stop

        stop1.terminal = True
        stop2.terminal = True
        stop3.terminal = True

        tscale = self.R/np.sqrt(2*self.E0/self.mu312)
        wsol = solve_ivp(y0 = self.w0, fun = lambda t,y: self.hamEq(t,y),
                         t_span = [0,tscale*self.t_stop], method = 'RK45',
                         rtol = self.r_tol, atol = self.a_tol, events = (stop1,stop2,stop3))

        self.wn = wsol.y
        self.t = wsol.t
        x = self.wn[:6] # rho1, rho2
        En, Vn, Kn, Ln = util.hamiltonian(self)
       
        self.delta_e = En[-1] - En[0] # Energy conservation
        self.delta_l = Ln[-1] - Ln[0] # Momentum conservation
        if self.delta_e > self.econs:
            print(f'Energy not conserved less than {self.econs}.')
            self.rejected+=1
            return
        if self.delta_l > self.lcons:
            print(f'Momentum not conserved less than {self.lcons}.')
            self.rejected+=1
            return
        
        r12,r23,r31 = util.jac2cart(x, self.C1, self.C2)

        # Recover vectors
        rho1x, rho1y, rho1z, rho2x, \
        rho2y, rho2z, p1x, p1y, p1z, \
        p2x, p2y, p2z = wsol.y

        # Components
        r23_x = rho2x - self.C1*rho1x
        r23_y = rho2y - self.C1*rho1y
        r23_z = rho2z - self.C1*rho1z
        r31_x = rho2x + self.C2*rho1x
        r31_y = rho2y + self.C2*rho1y
        r31_z = rho2z + self.C2*rho1z

        p23_x = self.mu23*p2x/self.mu312-self.mu23*p1x/self.m2
        p23_y = self.mu23*p2y/self.mu312-self.mu23*p1y/self.m2
        p23_z = self.mu23*p2z/self.mu312-self.mu23*p1z/self.m2
        p31_x = self.mu31*p2x/self.mu312+self.mu31*p1x/self.m1
        p31_y = self.mu31*p2y/self.mu312+self.mu31*p1y/self.m1
        p31_z = self.mu31*p2z/self.mu312+self.mu31*p1z/self.m1

        # Realtive momenta
        p12 = np.sqrt(p1x**2+p1y**2+p1z**2)
        p23 = np.sqrt(p23_x**2 + p23_y**2 + p23_z**2)
        p31 = np.sqrt(p31_x**2 + p31_y**2 + p31_z**2)

        # Angular momentum components
        j12_x = rho1y*p1z - rho1z*p1y
        j12_y = rho1z*p1x - rho1x*p1z
        j12_z = rho1x*p1y - rho1y*p1x

        j23_x = r23_y*p23_z - r23_z*p23_y
        j23_y = r23_z*p23_x - r23_x*p23_z
        j23_z = r23_x*p23_y - r23_y*p23_x        

        j31_x = r31_y*p31_z - r31_z*p31_y
        j31_y = r31_z*p31_x - r31_x*p31_z
        j31_z = r31_x*p31_y - r31_y*p31_x

        # j_eff arrays
        j12 = -0.5 + 0.5*np.sqrt(1 + 4*(j12_x**2 + j12_y**2 + j12_z**2))
        j23 = -0.5 + 0.5*np.sqrt(1 + 4*(j23_x**2 + j23_y**2 + j23_z**2))
        j31 = -0.5 + 0.5*np.sqrt(1 + 4*(j31_x**2 + j31_y**2 + j31_z**2))
        
        # Set j values
        self.mol_12.set_jPrime(np.round(j12)[-1])
        self.mol_23.set_jPrime(np.round(j23)[-1])
        self.mol_31.set_jPrime(np.round(j31)[-1])

        # Calculate internal energies (evib + erot)
        E12 = p12**2/2/self.mu12 + self.mol_12.Vij(r12) 
        E23 = p23**2/2/self.mu23 + self.mol_23.Vij(r23)
        E31 = p31**2/2/self.mu31 + self.mol_31.Vij(r31)

        # Set E attributes
        self.mol_12.set_E(E12[-1])
        self.mol_23.set_E(E23[-1])
        self.mol_31.set_E(E31[-1])
        
        # Set bound states
        self.mol_12.rebound()
        self.mol_23.rebound()
        self.mol_31.rebound()
        
        # Check if Eij < bound
        try:
            if self.mol_12.checkBound(r12[-1]):
                # Check for complex formation
                if not ((self.mol_23.checkBound(r23[-1])) or (self.mol_31.checkBound(r31[-1]))):
                    # If bound, set turning points and vprime
                    self.mol_12.turningPts(initial=False)
                    self.mol_12.gaussBin(j12[-1]) # Assign gaussian weight to (v,j)
                    self.count[0]+=1 # n12 
                    self.set_fstate((self.mol_12.vt,self.mol_12.vw,
                                self.mol_12.jPrime, self.mol_12.jw))
                else:
                    self.count[4]+=1 # complex
            elif self.mol_23.checkBound(r23[-1]):
                if not ((self.mol_31.checkBound(r31[-1])) or (self.mol_12.checkBound(r12[-1]))):
                    self.mol_23.turningPts(initial=False)
                    self.mol_23.gaussBin(j23[-1])
                    self.count[1]+=1 # n23
                    self.set_fstate((self.mol_23.vt,self.mol_23.vw,
                                self.mol_23.jPrime, self.mol_23.jw))
                else:
                    self.count[4]+=1 # complex
            elif self.mol_31.checkBound(r31[-1]):
                if not ((self.mol_12.checkBound(r12[-1])) or (self.mol_23.checkBound(r23[-1]))):
                    self.mol_31.turningPts(initial=False)
                    self.mol_31.gaussBin(j31[-1])
                    self.count[2]+=1 # n31
                    self.set_fstate((self.mol_31.vt,self.mol_31.vw,
                                self.mol_31.jPrime, self.mol_31.jw))
                else:
                    self.count[4]+=1 # complex
            else:
                self.count[3]+=1 # dissociation
        except:
            self.rejected = 1
def runOneT(*args,output=False,**kwargs):
    '''
    Runs one trajectory. Use this method as input into loop.
    '''
    input_dict = kwargs.get('input_dict')
    try:
        traj = Trajectory(**input_dict)
        traj.runT()
        res = util.get_results(traj,*args)
        if output:
            out = {k:[v] for k,v in res.items()} # turn scalar to list
            out = pd.DataFrame(out)
            out.to_csv(output,mode = 'a', index = False,
                    header = os.path.isfile(output) == False or os.path.getsize(output) == 0)
        return res
    except Exception as e:
        print(e)
        pass
    return

def runN(nTraj, input_dict, cpus = os.cpu_count(), attrs = None,
        short_out = None, long_out = None):
    t0 = time.time()
    result = []
    with mp.Pool(processes=cpus) as p:
        if attrs:
            event = [p.apply_async(runOneT, args = (*attrs,),kwds=({'output':long_out,'input_dict':input_dict})) for i in range(nTraj)]
        else:
            event = [p.apply_async(runOneT,kwds=({'output':long_out,'input_dict':input_dict})) for i in range(nTraj)]
        for res in event:
            result.append(res.get())
    result = [i for i in result if i is not None]
    full = pd.DataFrame(result)
    cols = ['vi','ji','e','b','n12','n23','n31','nd','nc']
    counts = full.loc[:,cols].groupby(['vi','ji','e','b']).sum() # sum counts
    counts['time'] = time.time() - t0
    # Short output
    if short_out:
        counts.to_csv(short_out, mode = 'a',
                    header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)
    
    return full, counts

if __name__ == '__main__':
    from potentials import *
    from constants import *
    m1 = 1.008*constants.u2me
    m2 = 1.008*constants.u2me
    m3 = 40.078*constants.u2me

    E0 = 40000 # collision energy (K)
    b0 = 0
    R0 = 50 # Bohr

    # Potential parameters in atomic units
    v12, dv12 = morse(de = 0.16456603489, re = 1.40104284795, alpha = 1.059493476908482)
    v23, dv23 = morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358)
    v31, dv31 = morse(de = 0.06529228457, re = 3.79079033313, alpha = 0.6906412379896358)

    v123, dv123dr12, dv123dr23, dv123dr31 = axilrod(C=0)

    # Define molecule dictionaries
    mol_12 = {'mi': m1, 'mj': m2, 'vi': 1, 'ji': 10, 'Vij':v12, 'dVij':dv12, 'xmin': .5, 'xmax': 30, 
        'npts':1000}
    mol_23 = {'mi': m2, 'mj': m3, 'Vij': v23, 'dVij': dv23, 'xmin': 1, 'xmax': 40}
    mol_31 = {'mi': m3, 'mj': m1, 'Vij': v31, 'dVij': dv31, 'xmin': 1, 'xmax': 40}

    # Initiate molecules
    mol12 = Molecule(mi = m1, mj = m2, vi = 1, ji = 0,Vij = v12, dVij = dv12, 
                        xmin = .5, xmax = 30, npts=1000)
    mol23 = Molecule(mi = m2, mj = m3, Vij = v23, dVij = dv23, xmin = 1, xmax = 40)
    mol31 = Molecule(mi = m3, mj = m1, Vij = v31, dVij = dv31, xmin = 1, xmax = 40)

    input_dict = {'m1':m1,'m2':m2,'m3':m3,
    'E0': E0, 'b0': b0, 'R0': R0, 'seed': None,
    'mol_12': mol12,'mol_23': mol23,'mol_31': mol31,
    'vt': v123, 'dvtdr12': dv123dr12, 'dvtdr23': dv123dr23, 'dvtdr31': dv123dr31,
    'integ':{'t_stop': 2, 'r_stop': 2, 'r_tol': 1e-12, 'a_tol':1e-10,'econs':1e-5,'lcons':1e-5}}


    ################################################
    import plotters
    bi = np.arange(100)
    input_dict['b0'] = 3.75
    input_dict['seed'] = 27
    traj = Trajectory(**input_dict)
    traj.runT()
    print(util.get_results(traj))
    plotters.traj_plt(traj)
    plt.show()
    # for b in bi:
    #     print(f'Running b={b}')
    #     input_dict['seed']=b
    #     runOneT(input_dict=input_dict)
    # traj = Trajectory(**input_dict)
    # traj.runT()
    # plotters.traj_plt(traj)
    # plt.show()
    # print(traj.mol_23.__dict__)
    # runN(30, input_dict, short_out='tryshort.txt', long_out='trylong.txt')
    # input_dict['seed'] = 63
    # traj = QCT(**input_dict)
    # traj.runT()
    # print(traj.delta_e)
    # plotters.traj_plt(traj)
    # plt.title(traj.count)
    # plt.show()
    # input_dict['E0'] = 40000
    # t0 = time.time()
    # nTraj = 20
    # runN(nTraj,input_dict, attrs=('delta_e',),long_out='long_test.txt', opacity='opac_test.txt', vib=False,rot=False)
    # print(f'Time: {time.time()-t0}')
    # n_jobs = 8
    # attrs = ('delta_e',)
    # r = Parallel(n_jobs=n_jobs)(delayed(runOneT)(*attrs,**input_dict) for i in range(nTraj))
    # df = pd.DataFrame(r)
    # print(df)
    # analysis.opacity(df,GB = False, vib = True, rot = False, output = 'opacity_test.txt', mode = 'a')
    ######## Test batch of trajectories ##########
    # from pyqcams2 import plotters
    # fig, axs = plt.subplots(2,5)
    # axs = axs.ravel()
    # for i in np.arange(0,10):    
    #     print(i)
    #     traj = Trajectory(**input_dict)
    #     try:
    #         traj.runT()
    #         plotters.traj_plt(traj, ax = axs[i])
    #         axs[i].set_title(f'{i}:{traj.count}')
    #     except Exception as e:
    #         print(e)
    #         pass
    # plt.show()