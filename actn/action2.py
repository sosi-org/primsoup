#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
actn: Simple Action Minimizer: Experiment 2

This script finds a tajectory that minimises the Action, given the start and end points.

It uses random local search (some local Monte Carlo) in space of trajectories.
It is not a particularly efficient algorithm.
It uses a crude and naïve/rudimentary/primitive representation of functionals.
'''


import numpy as np
import math
import matplotlib.pylab as pl
#import matplotlib

_KG, _Gr = 1.0, 0.001
_CM = 0.01


ntimesteps = 10*2
# segments of trajectory (nsteps = nsegments of time), discritisation segments of timespace trajectory

(X_AXIS, Y_AXIS) = (0, 1)
(XY_DIM, S_DIM) = (0, 1)
# x(s), y(s), t(s)
# s = (time)steps


#target_x_meters = 15.0 #* 2 #* 100.0 #    * 5.0 # *100
#target_y_meters = 1.0 #* 30
target_xy_meters = np.array([15.0, 1.0])
target_t_sec = 0.1  # 100 msec?!  Also corresponds to the index [-1] of clamp

def clamp1(traj):
    # clamp end points
    traj.xy[:,0] = (0.0, 0.0)
    traj.xy[:,-1] = target_xy_meters

    # a clammp constrsaint is also equivalent to an implicit force
    mi = int(traj.xy.shape[1]/2)
    traj.xy[:,mi] = (7.5, -1.0)
    #Also: get a bit closer to this

def clamp(traj):
    clamp1(traj)
    if False:
      traj2 = Trajectory(traj)
      clamp1(traj2)
      # get a bit closer to traj2
      alpha = 0.999
      traj.xy = traj.xy * (1-alpha) + traj2.xy * alpha

def generate_initial_path(ntimesteps):
    # not very sensitive to initial conditions
    return np.cumsum((np.random.randn(2,ntimesteps)+0.5)/(float(ntimesteps) * 0.5), axis=S_DIM) * target_xy_meters[:,None] # xy

class World:
    g = 9.8 * 1000.0

class Trajectory:
    def __init__(self, trajc=None, initial_path_xy=None):
        if trajc is None:
            self.xy = initial_path_xy.copy()
            self.m = 1.0 * _KG
            self.dt = target_t_sec / float(ntimesteps)
        else:
            self.xy = trajc.xy.copy()
            (self.m, self.dt) = (trajc.m, trajc.dt)

    def get_pot(self):
        h = self.xy[Y_AXIS,:]
        mgh = World.g * self.m  * h
        return np.sum(mgh) * self.dt  # integral = ∫ (mgh) dt

    def get_kin(self):
        v = np.diff(self.xy, axis=S_DIM) / self.dt
        mv2 = np.sum(0.5 * self.m * v * v, axis=XY_DIM)
        return np.sum(mv2) * self.dt  # integral =  ∫ (0.5 mv^2) dt

    def get_action(self):
        return self.get_kin() - self.get_pot()

def mutate(old_traj):
    # Candidate trajectory
    cand = Trajectory(old_traj)

    #############
    # Mutation site: @(s=j)
    ##############
    j = np.random.randint(0, old_traj.xy.shape[S_DIM])
    assert j >= 0
    assert j < ntimesteps
    # Don't mutate clamped positions:
    if j == 0:
        return None

    if j == cand.xy.shape[1]-1:
        return None

    ##############
    # Mutation
    ##############
    PERTURB = np.array([0.1, 0.02])
    uxy = (np.random.rand(2)*2-1.0)     # Normal(0,σ)
    #uxy = (np.random.randn(2))         # uniform [-1,1] * σ
    cand.xy[:,j] = cand.xy[:,j] + uxy * PERTURB[:]
    # todo: visualise (K,T), subtract (K'-K, T'-T)

    clamp(cand)

    return cand

#def accept_mutation(cand, old_traj):
#    return
def mouse_move(event):
    currentTraj.xy[:,-1] = [event.xdata, event.ydata]

def live_fig_update(currentTraj, i):
    handle = pl.plot(currentTraj.xy[X_AXIS,:], currentTraj.xy[Y_AXIS,:], 'b') #, 'color',(0.3,0.3,0.3) )
    handle[0].set_linewidth(0.2)
    pl.gca().set_aspect('equal')
    print(i)
    pl.connect('motion_notify_event', mouse_move)
    #pl.draw()
    pl.pause(0.001)
    print( currentTraj.get_action() )
    pl.show(block=False)


# hyper/meta trajectory
hyper_traj = []

PER_HOW_MANY = 500*2

seqoa=[]
seqoa_i=[]
accepted_count = 0
currentTraj = Trajectory(initial_path_xy=generate_initial_path(ntimesteps) )
clamp(currentTraj)

MAX_COUNT = int(210000)

for i in range(0,MAX_COUNT):
    sometimes = (i % PER_HOW_MANY == 0) or (i < PER_HOW_MANY and i % 20 == 0)
    if sometimes:
        hyper_traj.append((currentTraj.xy[:,:])[None,:,:])
        live_fig_update(currentTraj, i)

    # Candidate trajectory
    cand = mutate(currentTraj)
    if cand == None:
        continue # skip

    # Acceptance criteria
    action_new = cand.get_action()
    da = action_new - currentTraj.get_action()
    if da > 0: # Got worse (increased). We want the least action
        continue

    #Temp = 100.0
    #probr = math.exp( -abs(da)/Temp )
    #print probr

    # accept the mutation
    currentTraj = cand

    seqoa.append(action_new)
    seqoa_i.append(i)

    accepted_count += 1
    if accepted_count % PER_HOW_MANY == 0:
        print( currentTraj.get_action() )

def filter1(a, alpha):
    a = np.array(a); assert len(a.shape) == 1
    b = a.copy()
    slowa = a[0]
    for i in range(1, b.shape[0]):
        b[i] = slowa = slowa * (1.0-alpha) + a[i] * (alpha)
    return b

def overall_plot():
    ############
    # Plot overall indicators of trajectory of learning
    ############

    # What is this figure showing?
    # ta: not the real time, not the s: the learning time! τ
    # x-axis = Abscissa = slow time: τ
    # # not physical time
    ta=np.arange(0.0,float(len(seqoa)))/float(len(seqoa))

    Dτ=1.0 # not physical time, # 0.01

    fig, (ax1, ax2) = pl.subplots(1, 2)
    fig2(ax2)
    fig1(ax1)

def fig2(ax2):
    h0, = ax2.plot(ta,np.array(seqoa),'r', label='A')
    ax2b=ax2.twinx()
    h1, = ax2b.plot(ta[1:],np.diff(np.array(seqoa)), 'k.', markersize=0.2, label='ΔA')
    h2, = ax2b.plot(ta[1:],np.diff(filter1(seqoa, 0.01))/Dτ, 'b', label='dA')  # dx/dt

    fig2_annot([ax2, ax2b], [h0,h1,h2])


def fig2_annot(aa, hhh):
    [ax2, ax2b] = aa
    [h0,h1,h2] = hhh

    ax2.set_xscale('log')
    #ax2.set(xlabel='τ (epoc)', ylabel='A'); ax2.legend() # ax2.set_title('τ,A') # Action
    ax2b.set_ylim((-8.000, 0.2))
    ax2.set(xlabel='τ (epoc)');
    ax2.set_ylabel ('A = Action',  color='r')
    #ax2.yaxis.label.set_color(h0.get_color())
    ax2b.set_ylabel('ΔA', color='b') # ax2b.set(ylabel='ΔA')
    #ax2b.yaxis.label.set_color(h2.get_color())
    ax2.legend(handles=[h0, h1, h2], loc='lower center')
    #ax2b.spines.right.set_position(("outward", -10))
    ax2.spines.left.set_position(("outward", -30))
    ax2.spines.left.set_color('r')
    ax2b.spines.right.set_color('b')
    if False:
        #ax2.yticks(rotation = 45)
        ax2.spines.left.set_in_layout(True)
         # see https://matplotlib.org/stable/api/spines_api.html
        #ax2.spines.set_in_layout(True)
        ax2.set_in_layout(False)
        # https://matplotlib.org/3.5.0/api/transformations.html#matplotlib.transforms.Transform

        ##ax2.spines.left.set_transform(matplotlib.transforms.Affine2D.identity().rotate(5))
        ax2b.spines.left.set_transform(matplotlib.transforms.Affine2D.identity().rotate(0.2))


def fig1(ax1):
    # Plot certain streaks in the overall trajectory of learning
    xyz = np.concatenate(hyper_traj,axis=0)
    print(xyz.shape) #(:, 2, ntimesteps)
    ax1.plot(currentTraj.xy[X_AXIS,:], currentTraj.xy[Y_AXIS,:], 'k')
    ax1.set(xlabel='x', ylabel='y') #ax1.set_title('X,Y')
    # pl.hold(true)
    for ii in [3,5]: # out of ntimesteps
        ax1.plot(np.transpose(xyz[:,0,:]), np.transpose(xyz[:,1,:]), 'b-', linewidth=0.2)
        ax1.plot(xyz[-1,0,:], xyz[-1,1,:], 'b-', linewidth=0.4)
        ax1.plot(xyz[:,0,ii], xyz[:,1,ii], 'r.--')
        ax1.set(xlabel='x', ylabel='y') #pl.gca().set


overall_plot()

print('Finished. Close the plot. Press Q')
pl.show()
