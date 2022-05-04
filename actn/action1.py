#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
actn: Simple Action Minimizer

Based on the concept of "Action" defined on "trajectories".
Action and Lagragian are concepts defined by Joseph-Louis Lagrange
that are able to describe the whole classical mechanics.

This script finds a tajectory that minimises the Lagrange-ian Action,
for given start and end points.

It uses random local search (some local Monte Carlo) in space of trajectories.
It is not an efficient algorithm.
It uses a crude and naïve/rudimentary/primitive representation of functionals.
'''


import numpy as np
import math
import matplotlib.pylab as pl

_KG, _Gr = 1.0, 0.001
_CM = 0.01

# todo: xdim, ydim
target_x_meters = 5.0 #* 2 #* 100.0 #    * 5.0 # *100
target_y_meters = 1.0 #* 30
target_t_sec = 0.1  # 100 msec?!
ntimesteps = 10
#ntimesteps = 20  # todo: not stable / converging to the same value when ntimesteps (discritisation segments of timespace trajectory) is increased
#ntimesteps = 10+15 # segments of trajectory (nsteps = nsegments of time)
# = p.xy.shape[1]

(X_AXIS, Y_AXIS) = (0, 1)
(XY_DIM, S_DIM) = (0, 1)
# x(s), y(s), t(s)
# s = (time)steps

# Should reach y=100

class World:
    g = 9.8 * 10000.0

class Trajectory:
    def __init__(self, trajc=None):
        if not trajc is None:
            self.xy = trajc.xy.copy()
        else:
            # initial path
            self.xy = np.cumsum(np.random.rand(2,ntimesteps)/(float(ntimesteps) * 0.5), axis=S_DIM) * np.array([target_x_meters,target_y_meters])[:,None]

        self.m = 1.0 * 0.0001 * _KG
        self.dt= target_t_sec / float(ntimesteps)

    def get_pot(self):
        h = self.xy[Y_AXIS,:]
        mgh = World.g * self.m  * h
        return np.sum(mgh) * self.dt  # integral = ∫ (mgh) dt

    def get_kin(self):
        # diff: piecewise
        v = np.diff(self.xy, axis=S_DIM) / self.dt
        mv2 = np.sum(0.5 * self.m * v * v, axis=XY_DIM)
        return np.sum(mv2) * self.dt  # integral =  ∫ (0.5 mv^2) dt

    def get_action(self):
        return self.get_kin() - self.get_pot()

def mutate(old_traj):
    # candidate trajectory
    cand = Trajectory(p)
    #print p.get_pot()
    #print p.get_kin()

    #############
    # Mutation site: @(s=j)
    ##############
    #j = int(np.random.rand()*p.xy.shape[S_DIM])
    j = np.random.randint(0, p.xy.shape[S_DIM])
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
    PERTURB = 0.1
    # Normal(0,σ)
    dxy = (np.random.rand(2)*2-1.0)*PERTURB
    # uniform [-1,1] * σ
    #dxy = (np.random.randn(2))*PERTURB
    cand.xy[:,j] = cand.xy[:,j] + dxy
    #print p.get_pot(), p.get_kin()
    #print p.get_pot() - cand.get_pot(), p.get_kin() - cand.get_kin()

    return cand

def accept_mutation(cand, old_traj):
    return

# hyper/meta trajectory
hyper_traj = []

#PER_HOW_MANY = 500
PER_HOW_MANY = 50*10

seqoa=[]
seqoa_i=[]
accepted_count = 0
p=Trajectory()
# clamp end points
p.xy[:,0]=(0.0,0.0)
p.xy[:,-1]=(target_x_meters,target_y_meters)
#pl.plot(p.xy[0,:],p.xy[1,:], 'b')

MAX_COUNT = int(100000/2 * 1.4  * 10/10)
# MAX_COUNT =10000 # more brief, for debug

for i in range(0,MAX_COUNT):
    sometimes = i % PER_HOW_MANY == 0
    if sometimes:

        hyper_traj.append((p.xy[:,:])[None,:,:])

        handle = \
        pl.plot(p.xy[X_AXIS,:],p.xy[Y_AXIS,:], 'b') #, 'color',(0.3,0.3,0.3) )
        handle[0].set_linewidth(0.2)
        pl.gca().set_aspect('equal')
        print(i)
        #pl.draw()
        pl.pause(0.001)
        print( p.get_action() )
        pl.show(block=False)

    # candidate trajectory

    cand = mutate(p)
    if cand == None:
        # skip
        continue

    # Acceptance criteria

    action_new = cand.get_action()
    da = action_new - p.get_action()
    #print da
    if da > 0: # Got worse (increased). We want the least action
        continue

    #Temp = 100.0
    #probr = math.exp( -abs(da)/Temp )
    #print probr

    # accept the mutation
    p = cand

    seqoa.append(action_new)
    seqoa_i.append(i)

    accepted_count += 1
    if accepted_count % PER_HOW_MANY ==0:
        print( p.get_action() )

def filter1(a, alpha):
    a=np.array(a); print(a.shape)
    assert len(a.shape) == 1
    b = a.copy()
    slowa = a[0]
    for i in range(1, b.shape[0]):
        slowa = slowa * (1.0-alpha) + a[i] * (alpha)
        b[i] = slowa
    return b

############
# Plot overall indicators of trajectory of learning
############

# What is this figure showing?
# ta: not the real time, not the s: the learning time! τ
# x-axis = Abscissa = slow time: τ
# # not physical time
ta=np.arange(0.0,float(len(seqoa)))/float(len(seqoa))

fig, (ax1, ax2) = pl.subplots(1, 2)
# 0.01
DT=1.0 # not physical time

ax2.plot(ta,np.array(seqoa),'r', label='A')
ax2.plot(ta[1:],np.diff(np.array(seqoa))*1000, 'k.', markersize=0.2, label='ΔA')
ax2.plot(ta[1:],np.diff(filter1(seqoa, 0.01))/DT*1000, 'b', label='dA')  # dx/dt
ax2.set_xscale('log')
ax2.set(xlabel='τ', ylabel=None); ax2.legend() # ax2.set_title('τ,A') # Action

# Plot certain streaks in the overall trajectory of learning
xyz = np.concatenate(hyper_traj,axis=0)
print(xyz.shape) #(:, 2, ntimesteps)
ax1.plot(p.xy[X_AXIS,:],p.xy[Y_AXIS,:], 'k')
ax1.set(xlabel='x', ylabel='y') #ax1.set_title('X,Y')
#pl.figure()
# pl.hold(true)
for ii in [3,5]: # out of ntimesteps
    ax1.plot(np.transpose(xyz[:,0,:]), np.transpose(xyz[:,1,:]), 'b-', linewidth=0.2)
    ax1.plot(xyz[-1,0,:], xyz[-1,1,:], 'b-', linewidth=0.4)
    ax1.plot(xyz[:,0,ii], xyz[:,1,ii], 'r.--')
    ax1.set(xlabel='x', ylabel='y') #pl.gca().set

print('Finished. Close the plot. Press Q')
pl.show()
