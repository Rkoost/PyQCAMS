import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root_scalar, fsolve
from scipy import constants
import warnings
from . import util, constants
import json


class Energy(object):
    '''DVR with sine function basis for non-periodic functions over
        [xmin,xmax] interval.
    
    Inputs:
    mu, float
        reduced mass of the system in a.u.
    npts, int
        number of points
    xmin, xmax, int
        min and max of calculation interval
    '''
    def __init__(self, mu, npts, xmin = .1, xmax = 10., num_eigs = 30, j = 0):
        self.mu = mu
        self.npts = npts
        self.num_eigs = num_eigs
        self.xmin = xmin
        self.xmax = xmax
        self.xl = float(xmax)-float(xmin)
        self.dx = self.xl/npts
        self.n = np.arange(1,npts)
        self.x = float(xmin) + self.dx * self.n
        self.j = j

    def eDVR(self, V):
        '''Uses the DVR method to calculate initial vibrational energy level of H2
        V, function
            potential as a function of r
                Returns:
            list of discrete energy values
        '''
        VX = np.diag(V(self.x)) # Potential energy matrix
        _i = self.n[:,np.newaxis]
        _j = self.n[np.newaxis,:]
        m = self.npts + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore divide by 0 warn
            # Off-diagonal elements of kinetic energy matrix
            T = ((-1.)**(_i-_j)
                * (1/np.square(np.sin(np.pi/(2*m)*(_i-_j)))
                - 1/np.square(np.sin(np.pi/(2*m)*(_i+_j)))))
        # Diagonal elements of KE matrix
        T[self.n-1,self.n-1] = 0 
        T += np.diag((2*m**2+1)/3
             -1/np.square(np.sin(np.pi*self.n/m)))
        T *= np.pi**2/4/self.xl**2/self.mu
        HX = T + VX # Hamiltonian 
        evals, evecs = np.linalg.eigh(HX) # Solve the eigenvalue problem

        # To include the rotational coupling term 
        # E(v,j) = we*(v+.5) + wexe*(v+.5)^2 + bv*j*(j+1)
        # Where bv = be - ae*(v+.5) = (hbar^2/2m)<psi_v|1/r^2|psi_v>
        # We calculate the expectation value of the rotational energy of a 
        # vibrational eigenstate evecs
        # print((evecs[:,]**2).sum(axis=1)) # Check evecs is normalized
        bv = []
        for i in range(len(evecs)):
            bv.append(np.trapz(evecs[:,i]**2/(self.x**2),dx=self.dx)/2/self.mu/self.dx)
        bv = np.asarray(bv)
        bv *= self.j*(self.j+1)
        evj = evals + bv
        return evj
    
    def turningPts(self, V, re, v=0):
        ''' Setter function for turning points, period, & n of energy level of a given (v,j)
        Evj, list
            energy spectrum
        V, function
            potential
        mu, float
            reduced mass
        re, float
            equilibrium length of molecule
        v, int
            vibrational quantum number
        b, function
            buckingham potentential 
            (if V is piecewise, root_scalar doesn't work.
            However, DVR must still be done on the piecewise function.)
        '''
        self.v = v # Assign the molecules vibrational number
        self.evj = self.eDVR(V) 
        vbrot = lambda r: -self.j*(self.j+1)/2/self.mu/(r**2)- V(r)+ self.evj[self.v]
        # x = np.linspace(.5,10,1000)
        # plt.plot(x,V(x))
        # plt.hlines(evj[0], 0, 5)
        # plt.show()
        # print(self.xmin)
        rm = root_scalar(vbrot,bracket = [self.xmin,re]).root
        rp = root_scalar(vbrot,bracket = [re,self.xmax]).root
        
        # Integrate to solve for tau, n_vib
        tau = quad(lambda x: 1/np.sqrt(vbrot(x)),rm,rp)[0]
        vib = quad(lambda x: np.sqrt(vbrot(x)),rm,rp)[0]
        tau *= np.sqrt(2*self.mu)
        n_vib = np.round(-0.5 + vib*np.sqrt(2*self.mu)/np.pi)
        self.n_vib = n_vib
        self.tau = tau
        self.rm = rm
        self.rp = rp
        return 
    
    def get_rp(self):
        return self.rp

    def get_tau(self):
        return self.tau

    def get_j(self):
        return self.j

    def get_nvib(self):
        return self.n_vib
    

class QCT(object):
    '''Usage:

    QCT class for 3-body collisions (atom-molecule)
    Inputs (all in atomic units):
    m1, float
        Atomic mass of molecular atom 1
    m2, float
        Atomic mass of molecular atom 2         
    m3, float
        Atomic mass of free atom 3
    d0, float
        Initial distance between atom & molecule
    e0, float  
        Initial kinetic energy between atom & molecule
    b, float
        impact parameter
    mol, object
        initial molecule (H2)
    vi, function
        Potential of initial molecule (H2)
    vf, function
        Potential of final molecule (CaH)
    dvi, function
        Potential derivative of initial molecule (H2)
    dvf, function
        Potential derivative of final molecule (CaH)
    rmi, float
        classical inner turning-point of initial molecule
    rpi, float
        classical outer turning-point of initial molecule
    rmf, float
        classical inner turning-point of final molecule
    rpf, float
        classical outer turning-point of final molecule
    seed, int
        seed for random number generation to reproduce initial conditions
    '''
    def __init__(self,**kwargs):
        self.m1 = kwargs.get('m1')
        self.m2 = kwargs.get('m2')
        self.m3 = kwargs.get('m3')
        self.e0 = kwargs.get('e0')
        self.b = kwargs.get('b')
        self.d0 = kwargs.get('d0')
        self.tau = kwargs.get('mol').get_tau()
        self.j = kwargs.get('mol').get_j()
        self.n_vib = kwargs.get('mol').get_nvib()
        self.v1 = kwargs.get('v1')
        self.v2 = kwargs.get('v2')
        self.v3 = kwargs.get('v3')
        self.dv1 = kwargs.get('dv1') 
        self.dv2 = kwargs.get('dv2')
        self.dv3 = kwargs.get('dv3')
        self.rpi = kwargs.get('mol').get_rp()
        self.re1 = kwargs.get('re1')
        self.re2 = kwargs.get('re2')
        self.re3 = kwargs.get('re3')
        self.seed = kwargs.get('seed')
        self.t_stop = kwargs.get('int')['t_stop']
        self.far = kwargs.get('int')['r_stop']
        self.a_tol = kwargs.get('int')['a_tol']
        self.r_tol = kwargs.get('int')['r_tol']
        self.mtot = self.m1 + self.m2 + self.m3
        self.mu12 = self.m1*self.m2/(self.m1+self.m2) # H2
        self.mu31 = self.m1*self.m3/(self.m1+self.m3) # CaH_1
        self.mu32 = self.m2*self.m3/(self.m2+self.m3) # CaH_2
        self.mu123 = self.m3*(self.m1+self.m2)/self.mtot

    def vPrime(self, jeff, mu, V, dV, E, re):
        ''' Calculate the vibrational number of final state.
        jeff, int
            effective rotational quantum number
        mu, float
            reduced mass of the molecule
        V, function
            potential energy function of the molecule
        E, float
            energy level resulting from traj 
            (Ensure that the effective rotational energy was added)
        re, float
            equilibrium length of molecule
        '''
        vbrot = lambda r: -jeff*(jeff+1)/2/mu/(r**2)- V(r) + E # negative for large values of jeff
        # Find the new minimum to search for turning points nearby
        vp_eff = lambda r: dV(r) - jeff*(jeff+1)/mu/r**3 
        reNew = fsolve(vp_eff, re)
        # Solve turning points nearby the equilibrium distance for turning points
        # Check if rotational barrier is too large
        if vbrot(reNew) > 0: 
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                try: 
                    rm, rp = fsolve(vbrot,np.array([reNew*.9,reNew*1.1])) # Start small
                    vib = quad(lambda x: np.sqrt(vbrot(x)), rm, rp)[0]
                except Exception:
                    try:
                        rm, rp = fsolve(vbrot,np.array([reNew*.75,reNew*1.25])) # Medium
                        vib = quad(lambda x: np.sqrt(vbrot(x)), rm, rp)[0]
                    except Exception:
                        try: 
                            rm, rp = fsolve(vbrot, np.array([reNew*.5, reNew*1.5])) # Large jeff
                            vib = quad(lambda x: np.sqrt(vbrot(x)), rm, rp)[0]
                        except Exception:
                            vib = None
                            print(f'Turning point calculation failed. Initial conditions: \n \
                                d:{self.d};angles:{self.ang};n:{self.n_vib};j:{self.j}')
            # Multiply constants
            if vib != None:
                n_vib = -0.5 + vib*np.sqrt(2*mu)/np.pi
                return n_vib
            else:
                return vib
        # If rotational barrier is too high, the molecule dissociates
        else:
            vib = -1 
            return vib
        # vP = quad(vbrot, rm, rp)[0]
        # vP *= np.sqrt(2*mu)/np.pi
        # vP += -0.5
        # return np.round(vP)


    def iCond(self):
        ''' Generate initial conditions for system.
        b, float
            impact parameter
        d0, float
            initial distance between atom and molecule
        m1, m2, m3, mu12, mu123, float
            masses/reduced masses

        --------
        Returns Jacobi coordinates & conjugate momenta.
        '''
        # Momentum for relative coordinate
        p0 = np.sqrt(2*self.mu123*self.e0) 
        p1 = np.sqrt(self.j*(self.j+1))/self.rpi

        # Initial distance between atom and center of moleculeelf.d
        # tau is vibrational period of the initial molecule
        rng = np.random.default_rng(self.seed)
        self.d = self.d0 + rng.random()*self.tau*p0/self.mu123

        # Jacobi coordinates
        # Atom (relative coordinate)
        rho2x = 0
        rho2y = self.b # impact parameter
        rho2z = -np.sqrt(self.d**2-self.b**2)

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
        rho1x = self.rpi*np.sin(theta)*np.cos(phi)
        rho1y = self.rpi*np.sin(theta)*np.sin(phi)
        rho1z = self.rpi*np.cos(theta)

        # Conjugate momenta for internal coords rho1
        p1x = p1*(np.sin(phi)*np.cos(eta)
                - np.cos(theta)*np.cos(phi)*np.sin(eta))
        p1y = -p1*(np.cos(phi)*np.cos(eta)
                + np.cos(theta)*np.sin(phi)*np.sin(eta))
        p1z = p1*np.sin(theta)*np.sin(eta)

        return (rho1x,rho1y,rho1z,rho2x,rho2y,\
                rho2z,p1x,p1y,p1z,p2x,p2y,p2z)

    def hamiltonian(self, w, p):
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
            state parameters; p = [self.m1, self.m2, self.m3, self.mu12, self.mu23, self.mu31]
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
        rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, \
            p1x, p1y, p1z, p2x, p2y, p2z = w
        m1,m2,m3, mu12, mu23, mu31, mu123 = p 

        # Internuclear distances 
        r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)
        r32 = np.sqrt((rho2x + m2*rho1x/(m1+m2))**2
                    + (rho2y + m2*rho1y/(m1+m2))**2 
                    + (rho2z + m2*rho1z/(m1+m2))**2)
        r31 = np.sqrt((-rho2x + m1*rho1x/(m1+m2))**2
                    + (-rho2y + m1*rho1y/(m1+m2))**2 
                    + (-rho2z + m1*rho1z/(m1+m2))**2)
        
        # Kinetic energy
        ekin = 0.5*(p1x**2+p1y**2+p1z**2)/mu12 \
                + 0.5*(p2x**2+p2y**2+p2z**2)/mu123

        #Potential energy
        epot = self.v1(r12)+self.v2(r32)+self.v3(r31)
        # epot = H2_V(r12)
        # epot = 0
        
        etot = ekin + epot
        # Include rotational energy?

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

    def hamEq(self,t,w,p):
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
        m1,m2,m3, mu12, mu23, mu31, mu123 = p 
        # mtot = m1 + m2 + m3
        # mu123 = m3*(m1+m2)/mtot
        
        # Internuclear distances 
        r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)
        r32 = np.sqrt((rho2x - m1*rho1x/(m1+m2))**2
                    + (rho2y - m1*rho1y/(m1+m2))**2 
                    + (rho2z - m1*rho1z/(m1+m2))**2)
        r31 = np.sqrt((rho2x + m2*rho1x/(m1+m2))**2
                    + (rho2y + m2*rho1y/(m1+m2))**2 
                    + (rho2z + m2*rho1z/(m1+m2))**2)

        # Derivatives 
        r12drho1x = rho1x/r12
        r12drho1y = rho1y/r12
        r12drho1z = rho1z/r12
        r32drho1x = -m1*(rho2x-m1*rho1x/(m1+m2))/r32/(m1+m2)
        r32drho1y = -m1*(rho2y-m1*rho1y/(m1+m2))/r32/(m1+m2)
        r32drho1z = -m1*(rho2z-m1*rho1z/(m1+m2))/r32/(m1+m2)
        r32drho2x = (rho2x-m1*rho1x/(m1+m2))/r32
        r32drho2y = (rho2y-m1*rho1y/(m1+m2))/r32
        r32drho2z = (rho2z-m1*rho1z/(m1+m2))/r32
        r31drho1x = m2*(rho2x+m2*rho1x/(m1+m2))/r31/(m1+m2)
        r31drho1y = m2*(rho2y+m2*rho1y/(m1+m2))/r31/(m1+m2)
        r31drho1z = m2*(rho2z+m2*rho1z/(m1+m2))/r31/(m1+m2)
        r31drho2x = (rho2x+m2*rho1x/(m1+m2))/r31
        r31drho2y = (rho2y+m2*rho1y/(m1+m2))/r31
        r31drho2z = (rho2z+m2*rho1z/(m1+m2))/r31

        # Create f = (rho1x', rho1y', rho1z', rho2x', rho2y', rho2z', p1x', p1y', p1z', p2x', p2y', p2z')
        f = [p1x/mu12, p1y/mu12, p1z/mu12, p2x/mu123, p2y/mu123, p2z/mu123,
            -self.dv1(r12)*r12drho1x-self.dv2(r32)*r32drho1x-self.dv3(r31)*r31drho1x,
            -self.dv1(r12)*r12drho1y-self.dv2(r32)*r32drho1y-self.dv3(r31)*r31drho1y,
            -self.dv1(r12)*r12drho1z-self.dv2(r32)*r32drho1z-self.dv3(r31)*r31drho1z,
            -self.dv2(r32)*r32drho2x-self.dv3(r31)*r31drho2x,
            -self.dv2(r32)*r32drho2y-self.dv3(r31)*r31drho2y,
            -self.dv2(r32)*r32drho2z-self.dv3(r31)*r31drho2z]

        # No atom-molecule interaction
        # f = [p1x/mu12, p1y/mu12, p1z/mu12, p2x/mu123, p2y/mu123, p2z/mu123,
        #     -H2_Vdr(r12)*r12drho1x,
        #     -H2_Vdr(r12)*r12drho1y,
        #     -H2_Vdr(r12)*r12drho1z,
        #     0,0,0]
        return f 

    
    def runT(self, doplot = False):
        '''Run the trajectory at impact parameter b.
        b, float
            impact parameter (bohr)


        Returns: 
        Number of times scattering, reaction, dissociation occurs. 
        '''
        # Solve the ODE, output the energy
        w0 = self.iCond() # initial position/momentum vectors
        p = [self.m1, self.m2, self.m3, self.mu12,
             self.mu32, self.mu31, self.mu123]


        def stop1(t,w,p=p):
            '''Stop integration when r12 > "far" AU'''
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            m1,m2,m3, mu12, mu23, mu31, mu123 = p 
            r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)

            return r12 - self.far

        def stop3(t,w,p=p):
            '''Stop integration when r31 > "far" AU'''
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            m1,m2,m3, mu12, mu23, mu31, mu123 = p 
            r31 = np.sqrt((rho2x + m2*rho1x/(m1+m2))**2
                        + (rho2y + m2*rho1y/(m1+m2))**2 
                        + (rho2z + m2*rho1z/(m1+m2))**2)
            return r31 - self.far

        def stop2(t,w,p=p):
            '''Stop integration when r32 > "far" AU'''
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            m1,m2,m3, mu12, mu23, mu31, mu123 = p 
            r32 = np.sqrt((rho2x - m1*rho1x/(m1+m2))**2
                        + (rho2y - m1*rho1y/(m1+m2))**2 
                        + (rho2z - m1*rho1z/(m1+m2))**2)
            return r32 - self.far
        

        stop1.terminal = True
        stop2.terminal = True
        stop3.terminal = True

        # Equation parameters
        tscale = self.d/np.sqrt(2*self.e0/self.mu123)
        wsol = solve_ivp(y0 = w0, fun = lambda t, y: self.hamEq(t,y,p),
            t_span = [0,tscale*self.t_stop], method = 'RK45', 
            rtol = self.r_tol, atol = self.a_tol, events = (stop1,stop2,stop3))

        wn = wsol.y
        self.t = wsol.t

        En, Vn, Kn, Ln = self.hamiltonian(wn,p)

        # plt.figure(1)
        # plt.plot(t, En, marker = '.', label = 'En')
        # plt.plot(t, Ln, marker = '.', label = 'Ln')
        # plt.plot(t, Kn, marker = '.', label = 'Kn')
        # plt.plot(t, Vn, marker = '.', label = 'Vn')
        # plt.legend(loc = 'upper left')

        # # Calculate final energy
        # wf = wn[-1]
        # Ef,Lf= hamiltonian(wsol[-1], p)
        # print(wn.shape)
        
        # Recover vectors
        rho1x, rho1y, rho1z, rho2x, \
        rho2y, rho2z, p1x, p1y, p1z, \
        p2x, p2y, p2z = wsol.y

        # Internuclear distances 
        r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)
        r32 = np.sqrt((rho2x - self.m1*rho1x/(self.m1+self.m2))**2
                    + (rho2y - self.m1*rho1y/(self.m1+self.m2))**2 
                    + (rho2z - self.m1*rho1z/(self.m1+self.m2))**2)
        r31 = np.sqrt((rho2x + self.m2*rho1x/(self.m1+self.m2))**2
                    + (rho2y + self.m2*rho1y/(self.m1+self.m2))**2 
                    + (rho2z + self.m2*rho1z/(self.m1+self.m2))**2)

        # Components
        r32_x = rho2x - self.m1*rho1x/(self.m1+self.m2)
        r32_y = rho2y - self.m1*rho1y/(self.m1+self.m2)
        r32_z = rho2z - self.m1*rho1z/(self.m1+self.m2)
        r31_x = rho2x + self.m2*rho1x/(self.m1+self.m2)
        r31_y = rho2y + self.m2*rho1y/(self.m1+self.m2)
        r31_z = rho2z + self.m2*rho1z/(self.m1+self.m2)
        p32_x = self.mu32*p2x/self.mu123-self.mu32*p1x/self.m2
        p32_y = self.mu32*p2y/self.mu123-self.mu32*p1y/self.m2
        p32_z = self.mu32*p2z/self.mu123-self.mu32*p1z/self.m2
        p31_x = self.mu31*p2x/self.mu123+self.mu31*p1x/self.m1
        p31_y = self.mu31*p2y/self.mu123+self.mu31*p1y/self.m1
        p31_z = self.mu31*p2z/self.mu123+self.mu31*p1z/self.m1

        p12 = np.sqrt(p1x**2+p1y**2+p1z**2)
        p32 = np.sqrt(p32_x**2 + p32_y**2 + p32_z**2)
        p31 = np.sqrt(p31_x**2 + p31_y**2 + p31_z**2)

        p2 = np.sqrt(p2x**2 + p2y**2 + p2z**2)
        # Angular momentum components
        j12_x = rho1y*p1z - rho1z*p1y
        j12_y = rho1z*p1x - rho1x*p1z
        j12_z = rho1x*p1y - rho1y*p1x

        j32_x=p32_z*r32_y-p32_y*r32_z
        j32_y=p32_x*r32_z-p32_z*r32_x
        j32_z=r32_x*p32_y-r32_y*p32_x

        j31_x=p31_z*r31_y-p31_y*r31_z
        j31_y=p31_x*r31_z-p31_z*r31_x
        j31_z=r31_x*p31_y-r31_y*p31_x

        # j_eff arrays
        j12_eff = np.round(-0.5 + 0.5*np.sqrt(1+4*(j12_x**2+j12_y**2+j12_z**2)))
        j32_eff = np.round(-0.5 + 0.5*np.sqrt(1+4*(j32_x**2+j32_y**2+j32_z**2)))
        j31_eff = np.round(-0.5 + 0.5*np.sqrt(1+4*(j31_x**2+j31_y**2+j31_z**2)))

        # Calculate energies throughout trajectory
        E12 = p12**2/2/self.mu12 + self.v1(r12) 
        E32 = p32**2/2/self.mu32 + self.v2(r32) 
        E31 = p31**2/2/self.mu31 + self.v3(r31) 

        K12 = p12**2/2/self.mu12
        K32 = p32**2/2/self.mu32
        K31 = p31**2/2/self.mu31

    
        bd12 = util.bound(self.v1,self.dv1,j12_eff[-1],self.mu12, self.re1)
        bd32 = util.bound(self.v2,self.dv2,j32_eff[-1],self.mu32, self.re2)
        bd31 = util.bound(self.v3,self.dv3,j31_eff[-1],self.mu31, self.re3)

        if doplot == True:
            plt.figure()
            plt.plot(self.t, r12, label = 'r12')
            plt.plot(self.t, r32, label = 'r32')
            plt.plot(self.t, r31, label = 'r31')
            plt.title(f'Energy: {self.e0/constants.cEK2H}K')
            plt.xlabel('time (a.u.)')
            plt.ylabel('$r_(a_0)$')
            plt.legend()

            # plt.figure()
            # plt.plot(self.t,E12,label = 'E12')
            # plt.hlines(bd12,self.t[0],self.t[-1], linestyle = 'dashed',label = 'bd12')
            # plt.plot(self.t,E32,label = 'E32')
            # plt.hlines(bd32,self.t[0],self.t[-1], color = 'orange',linestyle = 'dashed', label = 'bd32')
            # plt.plot(self.t,E31,label = 'E31')
            # plt.hlines(bd31,self.t[0],self.t[-1], color = 'g',linestyle = 'dashed', label = 'bd31')
            # plt.hlines(0,self.t[0],self.t[-1])
            # # plt.plot(self.t, self.v1(r12), label = 'v12') # potential energy 
            # # plt.plot(self.t, self.v2(r32), label = 'v32')
            # # plt.plot(self.t, self.v3(r31), label = 'v31')
            # # plt.plot(self.t, K12, label = 'K12') # kinetic energy
            # # plt.plot(self.t, K32, label = 'K32')
            # # plt.plot(self.t, K31, label = 'K31')
            # plt.xlabel('time (a.u)')
            # plt.ylabel('$E_(E_H)$')
            # plt.legend()

            # plt.figure(3)
            # plt.plot(self.t, En) # Total energy
            plt.show()
        
        # Keep track of coordinates
        self.r = (rho1x,rho1y,rho1z,rho2x,rho2y,rho2z)
        # print(j31_eff)
        # x = np.linspace(.1,13,1000)
        # for i in range(10):
        #     plt.hlines(E31[-i],0,10, label = f'e31_{i}')
        # plt.plot(x,self.v3(x) + j31_eff[-1]*(j31_eff[-1]+1)/2/self.mu31/x**2)
        # plt.hlines(bd31, 0, 10, color = 'g')
        # plt.legend()
        # plt.show()
        # scrap if calculation ends before intermediate complex breaks apart
        n12,n32,n31,nd, comp = 0, 0, 0, 0, 0
        trash = 0
        vt = 0
        w = 0
        j_eff = 0
        # AB + C -> AB + C
        # Boundary is raised by centrifugal term 
        if E12[-1] < bd12:
            if E32[-1] > 0 and E31[-1] > 0: # Ensure an intermediate complex isn't formed
                # Find final (v, j)
                # Calculate the turning points of the new energy
                vp = self.vPrime(jeff = j12_eff[-1], mu = self.mu12, V = self.v1, dV = self.dv1, E = E12[-1], re = self.re1)
                if vp == None:
                    trash += 1
                elif vp == -1:
                    print(f'(12) New turning points could not be found. Check if Veff - E < 0 everywhere. Check that your \
                            "re" guess is less than the minimum of your potential. ')
                    nd += 1
                else:
                    vt = np.round(vp)
                    w = util.gaus(vp,vt)
                    # print(f'H2 final state: {vt,w,j12_eff}')
                    j_eff = j12_eff[-1]
                    n12 += 1
            else:
                comp += 1
        # AB + C -> BC + A
        elif E32[-1]  < bd32:
            if E12[-1] > 0 and E31[-1] > 0:
                vp = self.vPrime(jeff = j32_eff[-1], mu = self.mu32, V = self.v2, dV = self.dv2, E = E32[-1], re = self.re2)
                if vp == None:
                    trash += 1
                elif vp == -1:
                    print(f'(32) New turning points could not be found. Check if Veff - E < 0 everywhere. Check that your \
                            "re" guess is less than the minimum of your potential. ')
                    nd += 1
                else:                
                    vt = np.round(vp)
                    w = util.gaus(vp,vt)
                    # print(f'CaH(2) final state: {vt, w, j32_eff}')
                    j_eff = j32_eff[-1]
                    n32 += 1
            else:
                comp += 1
        # AB + C -> AC + B
        elif E31[-1]  < bd31:
            if E32[-1] > 0 and E12[-1] > 0:
                vp = self.vPrime(jeff = j31_eff[-1], mu = self.mu31, V = self.v3, dV = self.dv3, E = E31[-1], re = self.re3)
                if vp == None:
                    trash += 1
                elif vp == -1:
                    print(f'(31) New turning points could not be found. Check if Veff - E < 0 everywhere. Check that your \
                            "re" guess is less than the minimum of your potential. ')
                    nd += 1
                else:                
                    vt = np.round(vp)
                    w = util.gaus(vp,vt)
                    # print(f'CaH(1) final state: {vt, w, j31_eff}')
                    j_eff = j31_eff[-1]
                    n31 += 1
            else:
                comp += 1
        # AB + C -> A + B + C
        else:
            nd += 1

        np.set_printoptions(suppress = True) # Remove sci notation
        self.count = [n12,n32,n31,nd,comp]
        self.f_state = [vt, w, j_eff]
        self.f_p = [p1x[-1], p1y[-1], p1z[-1],
                    p2x[-1], p2y[-1], p2z[-1]]
        self.delta_e = En[-1]-En[0]
        self.delta_p = Ln[-1] - Ln[0]
        # return count, r, t
        # return count
        return

class Potential(object):
    """
    Potential functions.
    """
    def morse(self, de = 1.,alpha = 1.,re = 1., we = .1):
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
        we, float
            vibrational frequency
        '''
        V = lambda r: de*(1-np.exp(-alpha*(r-re)))**2 - de
        dV = lambda r: 2*alpha*de*(1-np.exp(-alpha*(r-re)))*np.exp(-alpha*(r-re))
        return V, dV
    
    def lj(self, m=12, n = 6, cm = 1., cn=1., **kwargs):
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
        V = lambda r: cm/r**(m)-n/r**(n)
        dV = lambda r: -m*cm/r**(m+1)+n*cn/r**(n+1)
        return V, dV

    def buckingham(self, a=1., b=1., c6 = 1., max = .1,**kwargs):
        '''Usage:
            V = buckingham(**kwargs)
        Buckingham potentials tend to come back down at low r. 
        We fix this by adding a piecewise wall.
        Return a one-dimensional Buckingham potential:
        V(r) = a*exp(-b*r) - c6/r^6 for r > r_x
        V(r) = 

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
        '''
        Buck = lambda r: a*np.exp(-b*r) - c6/r**6
        dBuck = lambda r: -a*b*np.exp(-b*r) + 6*c6/r**7
        ddBuck = lambda r: a*b**2*np.exp(-b*r) - 6*7*c6/r**8

        # Find the maximum of potential
        xi = fsolve(dBuck,max)
        # xi = fsolve(ddBuck,ip)
        # m =  dBuck(xi)
        # yi = Buck(xi)
        # Wall = lambda r:2*m*r + (yi - 2*m*xi)
        # dWall =  lambda r: 2*m

        # # Define piecewise nd_array within interpolation bounds
        # x = np.linspace(xmin,xmax,n)
        # Vx = np.piecewise(x,[x<xi,x>=xi], [Wall,Buck])
        # dVx = np.piecewise(x,[x<xi,x>=xi], [dWall,dBuck])

        # #Interpolate the nd array into a new function
        # V = interp1d(x,Vx)
        # dV = interp1d(x,dVx)
        
        return Buck, dBuck, xi
    
    

### make potentials ### 
def start(input_file):
    ''' Takes input data and makes the potential object.
    input_file, str
        directory to JSON input file
    '''
    with open(f'{input_file}') as f:
        data = json.load(f)
    potential_AB = data['potential_AB'] # which potential
    potential_BC = data['potential_BC']
    potential_CA = data['potential_CA']
    mol1 = data['potential_params']["AB"][f"{potential_AB}"]  # potential parameters
    mol2 = data['potential_params']["BC"][f"{potential_BC}"]
    mol3 = data['potential_params']["CA"][f"{potential_CA}"]
    xmin = data['potential_params']["AB"]['xmin'] 
    xmax = data['potential_params']["AB"]['xmax']
    m1,m2,m3 = data['masses'].values() # masses
    m12 = m1*m2/(m1+m2)
    m23 = m2*m3/(m2+m3)
    m31 = m1*m3/(m1+m3)
    vF = Potential()
    # Create energy object according to chosen potential
    if potential_AB == 'morse':
        mol1['alpha'] = mol1['we']*np.sqrt(m12/2/mol1['de'])
        mol1_V, mol1_dV = vF.morse(**mol1)
    elif potential_AB == 'lj':
        mol1_V, mol1_dV = vF.lj(**mol1)
    elif potential_AB == 'buck':
        mol1_V, mol1_dV, xmin= vF.buckingham(**mol1)

    AB = Energy(mu=m12, npts=data['potential_params']["AB"]['npts'], xmin = xmin,
                xmax = xmax, j = data['ji'])
    AB.turningPts(mol1_V,mol1['re'], v= data['vi']) # Assign attributies evals, rm, rp
    # else:
    #     print('This potential is not available: \
    #           Choose from morse, lj, buck.')

    if potential_BC == 'morse':
        mol2['alpha'] = mol2['we']*np.sqrt(m23/2/mol2['de'])
        mol2_V, mol2_dV = vF.morse(**mol2)
    elif potential_BC == 'lj':
        mol2_V, mol2_dV = vF.lj(**mol2)
    elif potential_BC == 'buck':
        mol2_V, mol2_dV, _ = vF.buckingham(**mol2)
    else:
        print('This potential is not available: \
              Choose from morse, lj, buck.')

    if potential_CA == 'morse':
        mol3['alpha'] = mol3['we']*np.sqrt(m31/2/mol3['de'])
        mol3_V, mol3_dV = vF.morse(**mol3)
    elif potential_CA == 'lj':
        mol3_V, mol3_dV = vF.lj(**mol3)
    elif potential_CA == 'buck':
        mol3_V, mol3_dV, _ = vF.buckingham(**mol3)
    else:
        print('This potential is not available: \
              Choose from morse, lj, buck.')

    # Calculate relevant attributes
    inputs = {'m1' : m1, 'm2': m2, 'm3': m3, 
         'd0' : data['r0'], 'e0' : data['Ec (K)']*constants.cEK2H,
         'b': data['b'],'mol': AB, 'v1' : mol1_V, 'v2' : mol2_V, 'v3':mol3_V,
         'dv1' : mol1_dV,'dv2': mol2_dV, 'dv3': mol3_dV,
         're1' : mol1['re'], 're2': mol2['re'], 're3': mol3['re'],
         'seed' : data['seed'], 'int': data['int_params']}
    return inputs

### Run a trajectory ###
def main(plot = False,**kwargs):
    a = QCT(**kwargs)
    a.runT(doplot = plot)
    # result = {'d_e': a.delta_e}
    # return result
    return util.get_results(a)

if __name__ == '__main__':
    # inputs = start('inputs.json')
    # # for i in range(10):
    # #     a = QCT(**inputs)
    # #     a.runT()
    # #     with open('e_cons.csv','a') as f:
    # #         f.write(f'{a.delta_e}\n')
    # print(main(plot = True, **inputs))
    print(constants.angstrom/constants.physical_constants['Bohr radius'][0])
