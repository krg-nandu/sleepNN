import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def run_simulation():

    delta_t = 0.05
    # set of three neurons
    r = np.zeros((3,1))

    R = np.zeros((3,1000))
    xs1, us1 = [], []
    xs2, us2 = [], []

    # weight matrix
    W = np.array([
                [0.8, 0,  0],
                [1,   0,  1],
                [0,   0,  0.8]
                ])

    _U = 0.45

    # init depressing synapses
    X = np.array([[0.,0,0], [0., 0, 1.], [0,0,0.]])
    U = np.array([[0.,0,0], [_U, 0, _U], [0,0,0.]])

    alphaD = delta_t * np.array([[-1.,-1.,-1.], [8., -1., 30.], [-1., -1., -1.]]) ** -1.
    alphaF = delta_t * np.array([[-1.,-1.,-1.], [1., -1., 2.], [-1., -1., -1.]]) ** -1.

    maskA = (alphaD < 0.).astype(np.float32)
    maskB = (alphaD >= 0.).astype(np.float32)


    for t in range(1000):
        if (t == 5//delta_t) or (t == 20//delta_t) : #and t%(20//delta_t)==0:
            # deliver a volley of activity to the input neuron
            r[1] += 10.
        xs1.append(X[1,0])
        us1.append(U[1,0])
        xs2.append(X[1,2])
        us2.append(U[1,2])

        r, X, U = run_sim_step(r, W, X, U, alphaD, alphaF, _U, maskA, maskB) #run_sim_step(x[t], u[t], alpha_std, alpha_stf, U, spike)
        R[:,t] = r.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(131)

    efficacy1 = [i*j/_U for (i,j) in zip(xs1,us1)]
    ax.plot(xs1)
    ax.plot(us1)
    ax.plot(efficacy1)
    ax.plot([5//delta_t, 5//delta_t], [0, 1.], 'r--')
    ax.plot([20//delta_t, 20//delta_t], [0, 1.], 'b--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Synaptic efficacy')

    ax = fig.add_subplot(132)
    efficacy2 = [i*j/_U for (i,j) in zip(xs2,us2)]
    ax.plot(xs2)
    ax.plot(us2)
    ax.plot(efficacy2)
    ax.plot([5//delta_t, 5//delta_t], [0, 1.], 'r--')
    ax.plot([20//delta_t, 20//delta_t], [0, 1.], 'b--')
    ax.set_yticks([])
    ax.set_xlabel('Time')

    plt.show(block=False)

    v1 = R[:,150] / np.linalg.norm(R[:,150])
    v2 = R[:,450] / np.linalg.norm(R[:,450])
    print(np.dot(v1,v2))

    #fig = plt.figure()
    ax = fig.add_subplot(133, projection='3d')
    ax.quiver((0, 0), (0, 0), (0, 0), (v1[0], v2[0]), (v1[1], v2[1]), (v1[2], v2[2]), color=((1,0,0), (0,0,1)))
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.])
    ax.set_zlim([0., 1.])

    # normalize the rates
    #import ipdb; ipdb.set_trace()
    X = R/np.linalg.norm(R,axis=0)[np.newaxis,:]
    ax.scatter(X[0,:], X[1,:], X[2,:], c='k', linewidth=0, alpha=0.2)

    ax.set_xlabel('R1')
    ax.set_ylabel('R2')
    ax.set_zlabel('R3')
 
    ax.grid(False)
    plt.savefig('ortho_demo.png')
    plt.show()

def run_sim_step(r, W, X, U, alphaD, alphaF, _U, maskA, maskB, dt=10., tau=100.):

    X = X + (alphaD*(1 - X) - np.matmul((dt/1000.)* U * X, r))
    U = U + (alphaF*(_U - U) + np.matmul((dt/1000.)* _U * (1 - U), r))

    X = X*maskB + maskA
    U = U*maskB + maskA

    X = np.minimum(1, np.maximum(0, X))
    U = np.minimum(1, np.maximum(0, U))

    dt_neur = dt/tau
    r = np.maximum(0, r*(1 - dt_neur)
                   + dt_neur * np.matmul((W*X*U).T, r)
                   + np.random.normal(0, 0.01, size = r.shape))

    #h *= suppress_activity
    return r, X, U


def create_stp_constants(synapse_type='std', delta_t=1):
        
    # synapses can either be stp or std 
    
    if synapse_type == 'std': # make them all depressing
        tau_f = 20 # in milliseconds
        tau_d = 50
        U = 0.45
            
    elif synapse_type == 'stf': # make them all facilitating
        tau_f = 1500
        tau_d = 200
        U = 0.15
        
    else:
        print('Wrong STP specification')
            
    # convert time constants into decay rates    
    alpha_std = delta_t/tau_d
    alpha_stf = delta_t/tau_f

    return alpha_std, alpha_stf, U


def main():
    run_simulation()

if __name__ == '__main__':
    main()
