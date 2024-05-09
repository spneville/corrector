import sys as sys
import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs

def lhc(model=None, n_samples=None, bounds=None, modes=None):
    
    # No. modes to sample
    n_modes_sampled = len(modes)

    # Sampled geometries
    X = np.zeros((n_samples, model.n_atoms, 3))
    
    # Do the sampling of the normal modes
    space     = Space(bounds)
    lhs       = Lhs(lhs_type='classic', criterion=None)
    q_sampled = lhs.generate(space.dimensions, n_samples)

    # Convert to Cartesian coordinates
    Q = np.zeros((model.n_modes))
    for i in range(n_samples):
        for k in range(n_modes_sampled):
            Q[modes[k]-1] = q_sampled[i][k]
            X[i,:,:] = (model.x0 + model.q2x @ Q).reshape((model.n_atoms,3))
    
    return X
