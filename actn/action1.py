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
target_x_meters = 5.0 #* 2 #* 100.0 #	* 5.0 # *100
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

class Trajectory:
	g = 9.8 * 10000.0
	def __init__(self, pth=None): #,A,B):
		#self.xy[:,0]=np.array[0]
		#self.xy[:,1]=[0]
		if not pth is None:
			self.xy =pth.xy.copy()
		else:
			#self.xy = np.zeros((2,ntimesteps))
			# initial path
			self.xy = np.cumsum(np.random.rand(2,ntimesteps)/(float(ntimesteps) * 0.5), axis=S_DIM) * np.array([target_x_meters,target_y_meters])[:,None]
			print(self.xy)

		self.m = 1.0 * 0.0001 * _KG
		#self.t = ?
		self.dt= target_t_sec / float(ntimesteps)

	def get_pot(self):
		h = self.xy[Y_AXIS,:]
		mgh = self.g * self.m  * h
		return np.sum(mgh) * self.dt  # integral = ∫ (mgh) dt

	def get_kin(self):
		# diff: piecewise
		v = np.diff(self.xy, axis=S_DIM) / self.dt
		mv2 = np.sum(0.5 * self.m * v * v, axis=XY_DIM)
		return np.sum(mv2) * self.dt  # integral =  ∫ (0.5 mv^2) dt

	def get_action(self):
		return self.get_kin() - self.get_pot()

# hyper/meta trajectory
hyper_traj = []

#PER_HOW_MANY = 500
PER_HOW_MANY = 50*10

slowda=0
seqa=[]
seqoa=[]
ctr=0
p=Trajectory()
# clamp end points
p.xy[:,0]=(0.0,0.0)
p.xy[:,-1]=(target_x_meters,target_y_meters)
#pl.plot(p.xy[0,:],p.xy[1,:], 'b')

MAX_COUNT = int(100000/2 * 1.4) * 10
# MAX_COUNT =10000 # more brief, for debug

for i in range(0,MAX_COUNT):
	if i % PER_HOW_MANY ==0:

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
	cand = Trajectory(p)
	#print p.get_pot()
	#print p.get_kin()
	j = int(np.random.rand()*p.xy.shape[S_DIM])
	#print j
	assert j >= 0
	assert j < ntimesteps
	if j==0:
		continue
	if j==cand.xy.shape[1]-1:
		continue

	##############
	# Mutation
	##############
	PERTURB = 0.001*100 * 0.5
	# Normal(0,σ)
	dxy = (np.random.rand(2)*2-1.0)*PERTURB
	# uniform [-1,1] * σ
	#dxy = (np.random.randn(2))*PERTURB
	cand.xy[:,j] = cand.xy[:,j] + dxy
	#print p.get_pot(), p.get_kin()
	#print p.get_pot() - cand.get_pot(), p.get_kin() - cand.get_kin()
	a=cand.get_action()
	da= a - p.get_action()
	#print da
	if da>0: #worse
		continue
	#Temp = 100.0
	#probr = math.exp( -abs(da)/Temp )
	#print probr
	#seqa.append(a)
	p = cand

	alpha=0.01
	slowda = slowda * (1.0-alpha) + a * (alpha)
	seqa.append(slowda)
	seqoa.append(a)

	ctr+=1
	if ctr % PER_HOW_MANY ==0:
		print( p.get_action() )
	#print( p.get_action() )

pl.figure()
# What is this figure showing?
#DT=0.01
DT=p.dt # sec
pl.plot(p.xy[X_AXIS,:],p.xy[Y_AXIS,:], 'k')
ta=np.arange(0.0,float(len(seqa)))/float(len(seqa))
pl.plot(ta,np.array(seqoa),'r')
pl.plot(ta[1:],np.diff(seqa)/DT *10)  # dx/dt
pl.plot(ta,np.array(seqa)*0.0,'k')


xyz = np.concatenate(hyper_traj,axis=0)
print(xyz.shape) #(:, 2, ntimesteps)
pl.figure()
# pl.hold(true)
for ii in [3,5]: # out of ntimesteps
    pl.plot(np.transpose(xyz[:,0,:]), np.transpose(xyz[:,1,:]), 'b-', linewidth=0.2)
    pl.plot(xyz[-1,0,:], xyz[-1,1,:], 'b-', linewidth=0.2)
    pl.plot(xyz[:,0,ii], xyz[:,1,ii], 'r.-')

print('Finished. Close the plot.')
pl.show()
