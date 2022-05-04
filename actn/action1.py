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



class Path:
	g = 9.8
	def __init__(self, pth=None): #,A,B):
		#self.xy[:,0]=np.array[0]
		#self.xy[:,1]=[0]
		if not pth is None:
			self.xy =pth.xy.copy()
		else:
			#self.xy = np.zeros((2,10))
			self.xy = np.cumsum(np.random.rand(2,10),axis=1)
		#self.A=np.zeros((2,))
		#self.B=np.zeros((2,))
		self.m=1
		#self.t = ?
		self.dt= 0.01


	def piecewise(self):
		#np.diff(self.xy[0,:])
		return np.diff(self.xy, axis=1)

	def get_pot(self):
		return np.sum(self.g * self.m * self.xy[1,:]) * self.dt

	def get_kin(self):
		v = self.piecewise()
		xy=np.sum(0.5 * self.m * v * v, axis=1) * self.dt
		return np.sum(xy) #x+y

	def get_action(self):
		return self.get_kin() - self.get_pot()

slowda=0
seqa=[]
seqoa=[]
ctr=0
p=Path()
p.xy[:,0]=(0,0)
p.xy[:,-1]=(5,1)
#pl.plot(p.xy[0,:],p.xy[1,:], 'b')
for i in range(0,100000/2):
	if i %500 ==0:
		pl.plot(p.xy[0,:],p.xy[1,:], 'b') #, 'color',(0.3,0.3,0.3) )
	cand = Path(p)
	#print p.get_pot()
	#print p.get_kin()
	j = int(np.random.rand()*p.xy.shape[1])
	#print j
	assert j>=0
	assert j<10
	if j==0:
		continue
	if j==cand.xy.shape[1]-1:
		continue

	PERTURB = 0.001*100
	dxy = (np.random.rand(2)*2-1)*PERTURB
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
	if ctr%500 ==0:
		print p.get_action()
	#print p.get_action()

DT=0.01
pl.plot(p.xy[0,:],p.xy[1,:], 'k')
ta=np.arange(0.0,float(len(seqa)))/float(len(seqa))
pl.plot(ta,np.array(seqoa),'r')
pl.plot(ta[1:],np.diff(seqa)/DT *10)
pl.plot(ta,np.array(seqa)*0.0,'k')
pl.show()
