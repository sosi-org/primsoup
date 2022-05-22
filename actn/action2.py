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
    #traj.xy[:,mi] = (7.5, -1.0)

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
            # (2 x N)
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
        T,V = self.get_kin(), self.get_pot()
        return T - V, T, V

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

def init_trends():
    trend = { }
    trend['seqoA'] = []  # for all accepted
    trend['seqoK'] = []  # for all accepted
    trend['seqoV'] = []  # for all accepted
    trend['seqoFulTrajXY'] = []  # for all plotted only? no!  trajectories for # for all accepted
    # aalternative: "tau"s for all "traj"
    return trend

def register_trend(trend, i, cand, action_new, newK, newV):
    ''' called for each single accepted candidate '''
    trend['seqoA'].append((i, action_new, newK, newV))
    trend['seqoK'].append((i, newK))
    trend['seqoV'].append((i, newV))

    trend['seqoFulTrajXY'].append(cand.xy)

def simulate():
    # hyper/meta trajectory
    hyper_traj = []  # less frequent
    trend = init_trends()

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
        action_new, newK, newV = cand.get_action()
        cA, cK, cV = currentTraj.get_action()
        da = action_new - cA
        if da >= 0: # Got worse (increased). We want the least action
            continue

        ######################
        # Accepted trajectory
        ######################

        #Temp = 100.0
        #probr = math.exp( -abs(da)/Temp )
        #print probr

        # accept the mutation
        currentTraj = cand
        #cA, cK, cV = currentTraj.get_action()

        register_trend(trend, i, cand, action_new, newK, newV)

        accepted_count += 1
        if accepted_count % PER_HOW_MANY == 0:
            print( 'Action,K,V=', action_new,newK,newV )

    return currentTraj, hyper_traj, trend


fig_live = pl.figure()
fig_live.gca().set_aspect('equal')
def live_fig_update(currentTraj, i):
    handle, = pl.plot(currentTraj.xy[X_AXIS,:], currentTraj.xy[Y_AXIS,:], 'b') #, 'color',(0.3,0.3,0.3) )
    handle.set_linewidth(0.2)

    def mouse_move0(event):
        # currentTraj.xy[:,-1]
        # print(event)
        if event.xdata is not None and event.ydata is not None:
            user_mouse_fixation[:] = [event.xdata, event.ydata]

    pl.connect('motion_notify_event', mouse_move0)
    #pl.draw()
    pl.pause(0.001)
    print( 'action:', currentTraj.get_action(), '  iteration', i)
    pl.show(block=False)

def setLiveMouseHighlighter(pl, last_h, ax1, fig1, τa, hyper_traj_list):
    #  sets the closure, for subsequent calls to `mouse_move1`
    # The closure contains: `last_h`, `τai`, etc.  last_h -> matplotlib.lines.Line2D

    def find_nearest(array, value):
      '''
        array[return] ~= value
      '''
      # by Demitri -- https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
      idx = np.searchsorted(array, value, side="left")
      print()
      if idx > 0 and \
          (
            idx == len(array) or
            math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])
          ):
          return idx-1 # array[idx-1]
      else:
          return idx # array[idx]

    def mouse_move1(event):
        if event.xdata is not None and event.ydata is not None:
            '''
            implicit arguments:
                last_h: defined in closure set by setLiveMouseHighlighter()
                    wrong: τa is used for plotting last_h ? no.
                    It has nothing to do with that.
                    last_h is more about output.

                τa It is used in trends plot


                # wonrg: τai: index in the number of accepted
                τai: [index: number of accepted], value: iteration index
                τa: [index= number of accepted], value= iteration index * Dτ = the (float) x-value used for plotting

                "Has the same index with":
                    hyper_traj_list[A]
                    τa[A]
            '''

            # idx = index (among) the number of accepted
            #  τa[idx] ~= event.xdata
            idx = find_nearest(τa, event.xdata)
            print('nearest accpted_idx=', idx, 'x\'=iteration=', τa[idx], 'delta=x-x\'=', event.xdata-τa[idx])
            if False:
                xyz = hyper_traj[idx] # (1, 2, ntimesteps)
                xy = xyz[0] # first dimension is always 1, an unnecessry dimension
                last_h[0].set_data( xy[X_AXIS,:], xy[Y_AXIS,:] )
                #fig1.canvas.draw()
                #fig1.show()
                #pl.pause(0.1)
                pl.draw()
            if True:
                #print('ll', len(hyper_traj_list)) # len()=20701 = number of accepted
                xy = hyper_traj_list[idx]
                print('>.>', xy.shape) # (2, ntimesteps)
                last_h[0].set_data( xy[X_AXIS,:], xy[Y_AXIS,:] )
                pl.draw()

    pl.connect('motion_notify_event', mouse_move1)

def overall_plot(bestTraj, hyper_traj, trend):
    ############
    # Plot overall indicators of trajectory of learning
    ############

    # What is this figure showing?
    # τa: not the real time, not the s: the learning time! τ
    # x-axis = Abscissa = slow time: τ
    # # not physical time
    #nτ = len(trend['seqoA'])
    #τa = np.arange(0.0,float(nτ))/float(nτ)

    # Dτ=1.0 # not physical time, # 0.01
    #Dτ = 1.0/float(nτ)
    Dτ = 1.0

    fig, (ax1, ax2) = pl.subplots(1, 2)
    # fig1: the trajectories plot
    (last_h, ax1) = \
    fig1(ax1,   bestTraj, hyper_traj)

    # fig2: the trends plot
    τa = \
    fig2(ax2, Dτ, trend)

    #setLiveMouseHighlighter(pl, last_h, ax1, fig, τai, hyper_traj)
    #setLiveMouseHighlighter(pl, last_h, ax1, fig, τai, np.array(trend['seqoA'])[:,0])
    setLiveMouseHighlighter(pl, last_h, ax1, fig, τa, trend['seqoFulTrajXY'])
    
    # np.array(trend['seqoA'])[:,0]

def filter1(a, alpha):
    ''' Used for plotting only
    '''
    a = np.array(a); assert len(a.shape) == 1
    b = a.copy()
    slowa = a[0]
    for i in range(1, b.shape[0]):
        b[i] = slowa = slowa * (1.0-alpha) + a[i] * (alpha)
    return b

def fig2(ax2, Dτ, trend):
    seqoA_2 = np.array(trend['seqoA'])  # shape=(,4)
    seqoK_2 = np.array(trend['seqoK'])
    seqoV_2 = np.array(trend['seqoV'])
    '''
      indices:
        i: every iteration = EPOC = τi (τai)
        a: every accepted,
        every PER_HOW_MANY iterations,
        every PER_HOW_MANY accepted

        τ: i * Dτ = τai * Dτ : what is actually plotted (although Dτ=1.0 always )
        # τa

      # I_EPOC was practically: a: every accepted
      τai: index: [every accepted] value: every iteration
    '''
    I_EPOC, I_ACTION, I_K, I_V = (0,1,2,3)

    τai = seqoA_2[:,I_EPOC] # need the same values as τa, but as index
    # accepted indices, value = iteration index
    τa = τai * Dτ
    #τa = seqoA_2[:,I_EPOC] * Dτ
    # τaix = np.arange(1,τa.shape[0]) * Dτ
    # τaix -> τai


    #τa = seqoA_2[:,I_PLOT_IDX] * Dτ
    #τaix = τa

    h0A, = ax2.plot(τa, seqoA_2[:,I_ACTION],'r', label='A=T-V')
    h0K, = ax2.plot(τa, seqoA_2[:,I_K],'g', label='T')
    h0V, = ax2.plot(τa, seqoA_2[:,I_V],'c', label='V')
    """
    τa__ = τa_ * Dτ
    ax2.plot(τa__, seqoK_2[:,1],'g:', label='K')
    ax2.plot(τa__, seqoV_2[:,1],'c:', label='V')
    """
    ax2b = ax2.twinx()
    h1, = ax2b.plot(τa[1:], -np.diff(seqoA_2[:,I_ACTION]), 'k.', markersize=0.2, label='ΔA')
    h2, = ax2b.plot(τa[1:], -np.diff(filter1(seqoA_2[:,I_ACTION], 0.01)), 'b', label='dA')  # dx/dt

    fig2_annot([ax2, ax2b], [[h0A,h0K,h0V],h1,h2])

    pl.figure
    pl.yscale('log')
    h1, = ax2b.plot(τa[1:], -np.diff(seqoA_2[:,I_ACTION]), 'k.', markersize=0.2, label='ΔA')
    h2, = ax2b.plot(τa[1:], -np.diff(filter1(seqoA_2[:,I_ACTION], 0.01)), 'b', label='dA')  # dx/dt
    # why repeated?
    return τa

def fig2_annot(aa, hhh):
    [ax2, ax2b] = aa
    [[h0A,h0K,h0V], h1,h2] = hhh
    ax2.set_xscale('log')
    #ax2.set(xlabel='τ (epoc)', ylabel='A'); ax2.legend() # ax2.set_title('τ,A') # Action
    #ax2b.set_ylim((-2.000, 0.2))
    ax2b.set_yscale('symlog')
    ax2.set(xlabel='τ (epoc)')
    ax2.set_ylabel ('A = Action',  color='r')
    #ax2.yaxis.label.set_color(h0A.get_color())
    ax2b.set_ylabel('ΔA', color='b') # ax2b.set(ylabel='ΔA')
    #ax2b.yaxis.label.set_color(h2.get_color())
    ax2.legend(handles=[h0A, h0K, h0V, h1, h2], loc='lower center')
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
    if True:
      pass

def fig1(ax1, bestTraj, hyper_traj):
    # Plot certain streaks in the overall trajectory of learning
    xyz = np.concatenate(hyper_traj,axis=0)
    print(xyz.shape) #(:, 2, ntimesteps)
    last_h = \
    ax1.plot(bestTraj.xy[X_AXIS,:], bestTraj.xy[Y_AXIS,:], 'k', linewidth=5)
    ax1.set(xlabel='x', ylabel='y') #ax1.set_title('X,Y')
    # pl.hold(true)
    for ii in [3,5]: # out of ntimesteps
        ax1.plot(np.transpose(xyz[:,0,:]), np.transpose(xyz[:,1,:]), 'b-', linewidth=0.2)
        ax1.plot(xyz[-1,0,:], xyz[-1,1,:], 'b-', linewidth=0.4)
        ax1.plot(xyz[:,0,ii], xyz[:,1,ii], 'r.--')
        ax1.set(xlabel='x', ylabel='y') #pl.gca().set
    return (last_h, ax1)

bestTraj, hyper_traj, trend  = simulate()
overall_plot(bestTraj, hyper_traj, trend)

print('Finished. Close the plot. Press Q')
pl.show()
