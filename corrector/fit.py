import sys as sys
import numpy as np
from dscribe.descriptors import SOAP
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import normalize
import ase.io
from ase import Atoms

def gpr(model=None, geoms=None, potential=None):

    n_geom = len(geoms)

    # Differences between the computed and model potentials
    delta = np.zeros((n_geom, model.n_states, model.n_states))
    for i in range(n_geom):
        X = np.reshape(geoms[i], (model.n_cart))
        Q = model.x2q @ (X - model.x0)
        delta[i,:,:] = potential[i] - model.potential(Q)

    # Construct the SOAP descriptors
    soap = SOAP(species=list(set(model.atoms)),
                periodic=False,
                r_cut=4,
                n_max=6,
                l_max=6,
                sigma=0.1,
                average='inner')

    
    formula = ''
    for label in model.atoms:
        formula += label
    
    confs = [Atoms(formula) for i in range(n_geom)]

    for i in range(n_geom):
        confs[i].set_positions(geoms[i])

    confs_train, confs_test, delta_train, delta_test = \
    train_test_split(confs, delta, train_size=0.80, random_state=42)

    n_test  = len(confs_test)
    n_train = len(confs_train)
    
    p_train = normalize(np.array([soap.create(i) for i in confs_train]))
    p_test  = normalize(np.array([soap.create(i) for i in confs_test]))

    # Construct the GPR models
    kernel = 10*RBF(1.1,(1e-5,1e5)) + WhiteKernel(1e-8, (1e-10,10))
    gpr            = []
    delta_train_ii = np.zeros((n_train))
    delta_test_ii  = np.zeros((n_test))
    for i in range(model.n_states):

        print('\n State ', i+1)

        for n in range(n_train):
            delta_train_ii[n] = delta_train[n,i,i]
        for n in range(n_test):
            delta_test_ii[n] = delta_test[n,i,i]
        
        gpr.append(GaussianProcessRegressor(kernel=kernel,
                                            normalize_y=True,
                                            optimizer='fmin_l_bfgs_b',
                                            n_restarts_optimizer=30))
        
            
        gpr[i].fit(p_train, delta_train_ii)

        print('\n', gpr[i].kernel_)

        delta_star = gpr[i].predict(p_test)

        diff = delta_test_ii - delta_star
        
        rmsd = np.sqrt(np.dot(diff, diff)/n_test)
        print('\nRMSD:', rmsd)
        
        
    return gpr
