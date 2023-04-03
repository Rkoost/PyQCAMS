import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from .constants import *

def traj_plt(traj, ax, title = True, legend = True):
    '''
    Plot trajectory of a QCT calculation.

    Input:
    traj, QCT object
    ax, axis labele
    '''
    r = traj.r
    t = traj.t*ttos*10e12 # Convert a.u. to ns 
    m1,m2,m3 = traj.m1, traj.m2, traj.m3
    mtot = m1 + m2 + m3

    r12 = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    r32 = np.sqrt((r[3] - m1*r[0]/(m1+m2))**2
                + (r[4] - m1*r[1]/(m1+m2))**2 
                + (r[5] - m1*r[2]/(m1+m2))**2)
    r31 = np.sqrt((r[3] + m2*r[0]/(m1+m2))**2
                + (r[4] + m2*r[1]/(m1+m2))**2 
                + (r[5] + m2*r[2]/(m1+m2))**2)
    ax.plot(t, r12, label = 'r12')
    ax.plot(t, r32, label = 'r32')
    ax.plot(t, r31, label = 'r31')
    ax.set_xlabel('$t (ns)$')
    ax.set_ylabel('$r (a_0)$')
    if legend == True:
        ax.legend()
    if title == True:
        ax.set_title(f'Energy: {traj.e0/cEK2H}K')

def plot_e(traj, ax):
    ax.plot(traj.t,traj.E12,label = 'E12')
    # plt.hlines(bd12,traj.t[0],traj.t[-1], linestyle = 'dashed',label = 'bd12')
    ax.plot(traj.t,traj.E32,label = 'E32')
    # plt.hlines(bd32,traj.t[0],traj.t[-1], color = 'orange',linestyle = 'dashed', label = 'bd32')
    ax.plot(traj.t,traj.E31,label = 'E31')
    # plt.hlines(bd31,traj.t[0],traj.t[-1], color = 'g',linestyle = 'dashed', label = 'bd31')
    plt.hlines(0,traj.t[0],traj.t[-1])
    ax.set_xlabel('time (a.u)')
    ax.set_ylabel('$E_(E_H)$')
    ax.legend()
    ax.set_title('Energy vs t')


def traj_3d(traj):
    r = traj.r
    m1,m2,m3 = traj.m1, traj.m2, traj.m3
    mtot = m1 + m2 + m3
    c1 = m1/(m1+m2)
    c2 = m2/(m1+m2)
    r1 = np.array([-c2*r[i] - m3/mtot*r[i+3] for i in range(0,3)]) #x,y,z for particle 1
    r2 = np.array([c1*r[i]-m3/mtot*r[i+3] for i in range(0,3)])
    r3 = np.array([(m1+m2)/mtot*r[i+3] for i in range(0,3)])


    ax = plt.axes(projection='3d')
    ax.plot(r1[0], r1[1], r1[2], 'g')
    ax.plot(r2[0], r2[1], r2[2], 'orange')
    ax.plot(r3[0], r3[1], r3[2], 'r')
    ax.scatter3D(r1[0][0], r1[1][0], r1[2][0],marker = 'o',color = 'g')
    ax.scatter3D(r2[0][0], r2[1][0], r2[2][0],marker = 'o',color = 'orange')
    ax.scatter3D(r3[0][0], r3[1][0], r3[2][0],marker = 'o',color = 'r')
    ax.scatter3D(r1[0][-1], r1[1][-1], r1[2][-1],marker = '^',color = 'g')
    ax.scatter3D(r2[0][-1], r2[1][-1], r2[2][-1],marker = '^',color = 'orange')
    ax.scatter3D(r3[0][-1], r3[1][-1], r3[2][-1],marker = '^',color = 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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
    
    r = traj.r
    t = traj.t
    m1,m2,m3 = traj.m1, traj.m2, traj.m3
    mtot = m1 + m2 + m3
    c1 = m1/(m1+m2)
    c2 = m2/(m1+m2)
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

#     ax.set_title('3D Test')
    ax.view_init(theta,phi)

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, len(t), fargs=(data, lines),
                                    interval=1, blit=False)
    plt.show()
    return ax, line_ani

