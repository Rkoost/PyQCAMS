import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from pyqcams.constants import *
from pyqcams import util

def traj_plt(traj, ax = None, title = True, legend = True):
    if ax == None:
        ax = plt.gca()
    x = traj.wn[:6]
    t = traj.t*ttos # a.u. to s
    r12, r23, r31 = util.jac2cart(x,traj.C1,traj.C2)
    ax.plot(t,r12, label='r12')
    ax.plot(t,r23, label='r23')
    ax.plot(t,r31, label='r31')
    ax.set_xlabel('$t (s)$')
    ax.set_ylabel('$r (a_0)$')
    if legend == True:
        ax.legend()
    if title == True:
        ax.set_title(f'{traj.E0/K2Har} K')
    return ax

def traj_3d(traj, ax = None):
    if ax == None:
        ax = plt.gca()
    x = traj.wn[:6]
    r1 = np.array([-traj.C2*x[i] - traj.m3/traj.mtot*x[i+3] for i in range(0,3)])
    r2 = np.array([traj.C1*x[i]-traj.m3/traj.mtot*x[i+3] for i in range(0,3)])
    r3 = np.array([(traj.m1 + traj.m2)/traj.mtot*x[i+3] for i in range(0,3)])

    ax = plt.axes(projection='3d')
    ax.plot(r1[0], r1[1], r1[2], 'g', label = 'F1')
    ax.plot(r2[0], r2[1], r2[2], 'orange', label = 'F2')
    ax.plot(r3[0], r3[1], r3[2], 'r', label = 'F3')
    ax.scatter3D(r1[0][0], r1[1][0], r1[2][0],marker = 'o',color = 'g')
    ax.scatter3D(r2[0][0], r2[1][0], r2[2][0],marker = 'o',color = 'orange')
    ax.scatter3D(r3[0][0], r3[1][0], r3[2][0],marker = 'o',color = 'r')
    ax.scatter3D(r1[0][-1], r1[1][-1], r1[2][-1],marker = '^',color = 'g')
    ax.scatter3D(r2[0][-1], r2[1][-1], r2[2][-1],marker = '^',color = 'orange')
    ax.scatter3D(r3[0][-1], r3[1][-1], r3[2][-1],marker = '^',color = 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    return ax

def traj_gif(traj, theta, phi):
    '''
    Create an animation of a trajectory.
    Inputs:
    traj, trajectory object
    theta, initial viewing angle theta
    phi, initial viewing angle phi
    '''
    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines
    
    r = traj.wn[:6]
    t = traj.t
    m1,m2,m3 = traj.m1, traj.m2, traj.m3
    mtot = m1 + m2 + m3
    c1,c2 = traj.C1, traj.C2
    r1 = np.array([-c2*r[i] - m3/mtot*r[i+3] for i in range(0,3)]) #x,y,z for particle 1
    r2 = np.array([c1*r[i]-m3/mtot*r[i+3] for i in range(0,3)])
    r3 = np.array([(m1+m2)/mtot*r[i+3] for i in range(0,3)])
    data = [r1,r2,r3]

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
    # line = [ax.plot(r1[0],r1[1],r1[2])[0]]

    # Setting the axes properties
    ax.set_xlim3d([min(min(r1[0]),min(-r2[0]),min(r3[0]))/2, max(max(r1[0]),max(r2[0]),max(r3[0]))])
    ax.set_xlabel('X')

    ax.set_ylim3d([min(min(r1[1]),min(r2[1]),min(r3[1]))/2, max(max(r1[1]),max(r2[1]),max(r3[1]))])
    ax.set_ylabel('Y')

    ax.set_zlim3d([min(min(r1[2]),min(r2[2]),min(r3[2]))/2, max(max(r1[2]),max(r2[2]),max(r3[2]))])
    ax.set_zlabel('Z')

    # ax.set_title('3D Test')
    ax.view_init(theta,phi)

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, len(t), fargs=(data, lines),
                                    interval=1, blit=False)
    plt.show()
    return ax, line_ani