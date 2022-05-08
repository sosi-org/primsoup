#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
actn: Simple Action Minimizer: Experiment 2

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


user_mouse_fixation = []

#target_x_meters = 15.0 #* 2 #* 100.0 #    * 5.0 # *100
#target_y_meters = 1.0 #* 30
target_xy_meters = np.array([15.0, 1.0])
target_t_sec = 0.1 *40 # 100 msec?!  Also corresponds to the index [-1] of clamp

def clamp(traj):
    # clamp end points
    traj.xy[:,0] = (0.0, 0.0)
    dt = traj.dt
    # print(traj.xy[:,0].shape, np.array((1.0, 1.0)).shape, '**')
    # (2,) == (2,)
    # velocity xy
    vxy = np.array((-1.0, 1.0))
    traj.xy[:,1] = traj.xy[:,0] + vxy * dt

    traj.xy[:,-1] = target_xy_meters

    # a clamp constrsaint is also equivalent to an implicit force
    mi = int(traj.xy.shape[1]/2)
    traj.xy[:,mi] = (7.5, -1.0)

    # does not work properly
    if len(user_mouse_fixation) != 0:
        traj.xy[:,-1] = user_mouse_fixation


def generate_initial_path(ntimesteps):
    # not very sensitive to initial conditions
    return np.cumsum((np.random.randn(2,ntimesteps)+0.5)/(float(ntimesteps) * 0.5), axis=S_DIM) * target_xy_meters[:,None] # xy


'''
# failed
#get a bit closer to this (milder constraint force => won't work!)
def clamp(traj):
    traj2 = Trajectory(traj)
    clamp1(traj2)
    # get a bit closer to traj2
    alpha = 0.999
    traj.xy = traj.xy * (1-alpha) + traj2.xy * alpha
'''

class World:
    g = np.array([0, -9.8])

class Trajectory:
    def __init__(self, trajc=None, initial_path_xy=None):
        if trajc is None:
            self.xy = initial_path_xy.copy()
            self.m = 1.0 * _KG
            self.dt = target_t_sec / float(ntimesteps)
            """ Even $dt$ does not have to be uniform.
            It can be any $t(s)$.
            However, the integration is A = ∫ L dt
            In terms of s: A = ∫ L(s) dt(s)/ds ds.
            In terms of numerical computing, an array of `t` can be used,
            and can be stretched during the "evoluton" of trajectory (over τ ! homotopy?).
            In terms of Physics, time can be an emergent: read out from the trajectory.
            The trajectory itself is (transcended to be) in terms of (a in space of) s,τ.
            t(s,τ),
            t(s;τ), x(s;τ), y(s;τ), px(s;τ), py(s;τ)
            [t,x,y,px,py](s;τ)

            Why we don't store (x,px,py)? We store xy.
            The t, we know: t is (implcitly) simply: t(s) = s dt, s=0,1,...,ntimesteps.
            > Side note: `ntimesteps` is steps of `s`, not "time".
            But why don't we save (px,py)?
            Although we can extract it from the `xy` array,
            but in principle, it should have been "stored" separately.
            but the story is different:
            the correct interpretation is that `xy` and `(px,py)` are tightly constrained.
            Compelled to follow eachother.
            That's why we ... (?).
            However, it failed. A continuity constriant is somehow in place.
            Equivalent to what strength of force?

            The acceleration. The acceleration should be limited (constrined) now.

            Somehow $T$ needs to also take acceleration into account.
            This formula $T=Ek=0.5mv^2$ is somehow in absence of external (injected) force.

            Wikipedia says "no single expression for all physical systems".
            Is L for closed systems? (Exchange of forces between particles)
            However, if an external force is applied, the system is no longer closed.
            Is such cases, can we say Lagrangian doe snot exist?
            Perhaps the L=T-V is for free moving, when noexternal lforce is applied?

            If such L (for a noon-closed system) exists, a subsystem has L.
            For the same reasoon/logical symmetry, the other part of the system aalso has its own L'.
            What are the L of two interacting systems?
            Howe are they related?
            If a closed system is split into two parts two Ls will emerge.
            How an L can be divided into two L1,L2?

            It seems, L=L1+L2. (the Ti-Vi s add up).

            Wow ""The potential energy of the system reflects the energy of interaction between the particles, i.e. how much energy any one particle will have due to all the others and other external influences.".
            Yes, V from gravity, is from external force.
            That's the only force!

            Various force caategories:
            "conservative forces"
            For conservative "forces", V is a function of positions only: $V(x)$.
            "non-conservative forces" (have velocity)
            However, for "those non-conservative forces", the velocities will appear also: $V(x,p)$.
            V = field? Potential of (e.g. Electrical) field?

            "relativistic (forces)"
            wp: In relativistic systems (special and general),
            "dissiptive forces"
            also for "dissipative forces", the form of L will change.

            "Lagrange-multiplies forces" (reaction, rigid surface, intracting)
            Todo: The relation between L of interacting systems.

            "holonomic constraints":
            "holonomic constraints": $c(x,t)=0$.
            then ...?
            """
        else:
            self.xy = trajc.xy.copy()
            (self.m, self.dt) = (trajc.m, trajc.dt)

    def get_pot(self):
        gh =  np.sum(World.g[:,None] *  self.xy[:2,:], axis=XY_DIM)  # inner product g⋅x
        mgh = - self.m * gh
        return np.sum(mgh) * self.dt  # integral = ∫ (mgh) dt

    def get_kin(self):
        v = np.diff(self.xy, axis=S_DIM) / self.dt
        mv2 = np.sum(0.5 * self.m * v * v, axis=XY_DIM)
        # Somehow needs to also take acceleration into account.
        # This formula `Ek=0.5mv^2`` is somehow in absence of external (injected) force
        return np.sum(mv2) * self.dt  # integral =  ∫ (0.5 mv^2) dt

    def get_action(self):
        return self.get_kin() - self.get_pot()

def mutate(old_traj, actr):
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
    ss = 1.0 # math.exp(-actr*0.0001) * 10
    PERTURB = np.array([0.1, 0.02]) * ss
    uxy = (np.random.rand(2)*2-1.0)     # Normal(0,σ)
    #uxy = (np.random.randn(2))         # uniform [-1,1] * σ
    cand.xy[:,j] = cand.xy[:,j] + uxy * PERTURB[:]
    # todo: visualise (T,V), subtract (T'-T, V'-V)
    # Instead of the $(K,T)$ convension (as in $L = K-T$), I used (K,T).
    clamp(cand)

    return cand

#def accept_mutation(cand, old_traj):
#    return

def simulate():
    # hyper/meta trajectory
    hyper_traj = []  # less frequent
    trend = { }
    trend['seqoa'] = []  # for all accepted

    PER_HOW_MANY = 500*2

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
        cand = mutate(currentTraj, accepted_count)
        if cand == None:
            continue # skip

        # Acceptance criteria
        action_new = cand.get_action()
        da = action_new - currentTraj.get_action()
        if da > 0: # Got worse (increased). We want the least action
            continue

        ######################
        # Accepted trajectory
        ######################

        #Temp = 100.0
        #probr = math.exp( -abs(da)/Temp )
        #print probr

        # accept the mutation
        currentTraj = cand

        trend['seqoa'].append((i, action_new))
        accepted_count += 1
        if accepted_count % PER_HOW_MANY == 0:
            print( 'Action=', currentTraj.get_action() )

    return currentTraj, hyper_traj, trend

def filter1(a, alpha):
    a = np.array(a); assert len(a.shape) == 1
    b = a.copy()
    slowa = a[0]
    for i in range(1, b.shape[0]):
        b[i] = slowa = slowa * (1.0-alpha) + a[i] * (alpha)
    return b


fig_live = pl.figure()
fig_live.gca().set_aspect('equal')
def live_fig_update(currentTraj, i):
    handle, = pl.plot(currentTraj.xy[X_AXIS,:], currentTraj.xy[Y_AXIS,:], 'b') #, 'color',(0.3,0.3,0.3) )
    handle.set_linewidth(0.2)

    def mouse_move(event):
        # currentTraj.xy[:,-1]
        # print(event)
        if event.xdata is not None and event.ydata is not None:
            user_mouse_fixation[:] = [event.xdata, event.ydata]

    pl.connect('motion_notify_event', mouse_move)
    #pl.draw()
    pl.pause(0.001)
    print( 'action:', currentTraj.get_action(), '  iteration', i)
    pl.show(block=False)

def overall_plot(bestTraj, hyper_traj, trend):
    ############
    # Plot overall indicators of trajectory of learning
    ############

    # What is this figure showing?
    # τa: not the real time, not the s: the learning time! τ
    # x-axis = Abscissa = slow time: τ
    # # not physical time
    nτ = len(trend['seqoa'])
    τa = np.arange(0.0,float(nτ))/float(nτ)

    Dτ=1.0 # not physical time, # 0.01

    fig, (ax1, ax2) = pl.subplots(1, 2)
    fig2(ax2, τa, Dτ, trend['seqoa'])
    fig1(ax1,   bestTraj, hyper_traj)

def fig2(ax2, τa, Dτ, seqoa):
    seqoa_2 = np.array(seqoa)
    EPOC_I, ACTION_I = (0, 1)
    # print(seqoa_2.shape) # (2442, 2)
    h0, = ax2.plot(τa, seqoa_2[:,ACTION_I],'r', label='A')
    ax2b = ax2.twinx()
    h1, = ax2b.plot(τa[1:], np.diff(seqoa_2[:,ACTION_I]), 'k.', markersize=0.2, label='ΔA')
    h2, = ax2b.plot(τa[1:], np.diff(filter1(seqoa_2[:,ACTION_I], 0.01))/Dτ, 'b', label='dA')  # dx/dt

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


def fig1(ax1, bestTraj, hyper_traj):
    # Plot certain streaks in the overall trajectory of learning
    xyz = np.concatenate(hyper_traj,axis=0)
    print(xyz.shape) #(:, 2, ntimesteps)
    ax1.plot(bestTraj.xy[X_AXIS,:], bestTraj.xy[Y_AXIS,:], 'k')
    ax1.set(xlabel='x', ylabel='y') #ax1.set_title('X,Y')
    # pl.hold(true)
    for ii in [3,5]: # out of ntimesteps
        ax1.plot(np.transpose(xyz[:,0,:]), np.transpose(xyz[:,1,:]), 'b-', linewidth=0.2)
        ax1.plot(xyz[-1,0,:], xyz[-1,1,:], 'b-', linewidth=0.4)
        ax1.plot(xyz[:,0,ii], xyz[:,1,ii], 'r.--')
        ax1.set(xlabel='x', ylabel='y') #pl.gca().set

bestTraj, hyper_traj, trend  = simulate()
overall_plot(bestTraj, hyper_traj, trend)

print('Finished. Close the plot. Press Q')
pl.show()
