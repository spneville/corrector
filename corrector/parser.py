import sys as sys
import numpy as np
import os as os
import corrector.constants as constants

def parse_graci(model=None, directory=None):

    # Get the list of all filenames in the given
    # directory
    files = os.listdir(directory)
    paths = [directory+'/'+files[i] for i in range(len(files))]

    # Get the reference geometry ground state energy
    eref = read_eref(paths[0])
    
    # Parse the diabatic potential matrix elements and Cartesian
    # geometries
    X = []
    W = []
    for path in paths:
        W.append(read_diabpot(path, model.n_states, eref))
        X.append(read_geom(path, model.n_atoms))

    return X, W

def read_eref(path):

    eref = 0.

    search_string = 'gvvpt2:'

    with open(path, 'r') as f:
        lines = f.readlines()
        positions = []
        for i, line in enumerate(lines, 1):
            if search_string in line:
                positions.append(i-1)

        eref = float(lines[positions[-1]+10].split()[3])
        
    return eref

def read_diabpot(path, n_states, eref):

    n_elements = int(n_states * (n_states + 1) / 2)

    W = np.zeros((n_states, n_states))

    search_string = 'Diabatic potential matrix elements'
    
    with open(path, 'r') as f:
        lines = f.readlines()
        positions = []
        for i, line in enumerate(lines, 1):
            if search_string in line:
                positions.append(i-1)
        for i in range(positions[-1]+4, positions[-1]+4+n_elements):
            I = int(lines[i].split()[0])
            J = int(lines[i].split()[1])
            W[I-1,J-1] = float(lines[i].split()[2])
            W[J-1,I-1] = W[I-1,J-1]

    for i in range(n_states):
        W[i,i] -= eref
    W *= constants.eh2ev
            
    return W

def read_geom(path, n_atoms):

    X = np.zeros((n_atoms, 3))

    search_string = 'DFT/MRCI(2) computation'

    with open(path, 'r') as f:
        lines = f.readlines()
        positions = []
        for i, line in enumerate(lines, 1):
            if search_string in line:
                positions.append(i-1)
        n = -1
        for i in range(positions[-1]+6, positions[-1]+6+n_atoms):
            n += 1
            X[n,0] = float(lines[i].split()[1])
            X[n,1] = float(lines[i].split()[2])
            X[n,2] = float(lines[i].split()[3])

    return X
