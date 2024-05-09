import sys as sys
import numpy as np
import os as os
import shutil as shutil

def write_geoms(model=None, X=None, delta=0.5):

    # Make the output directory
    if os.path.exists('./geoms'):
        shutil.rmtree('./geoms')
    os.mkdir('./geoms')
        
    # Write the paths to the sampled geometries
    fmat = '{:3} {:10.6f} {:10.6f} {:10.6f}\n'
    Qj = np.zeros((model.n_modes), dtype=float)
    for i in range(X.shape[0]):

        xi = X[i,:,:].reshape(model.n_cart)
        Q  = model.x2q @ (xi - model.x0)

        length  = np.linalg.norm(Q)
        n_steps = int(np.ceil(length / 0.5))

        dQ = Q / n_steps

        with open('./geoms/geom'+str(i+1)+'.xyz', 'w') as f:
            for j in range(n_steps+1):
                Qj = j * dQ
                Qj = j * dQ
                xj = model.x0 + model.q2x @ Qj
                xj = xj.reshape((model.n_atoms, 3))
                f.write(str(model.n_atoms)+'\n\n')
                for k in range(model.n_atoms):
                    f.write(fmat.format(model.atoms[k],
                                        xj[k,0],
                                        xj[k,1],
                                        xj[k,2]))
                
    return
