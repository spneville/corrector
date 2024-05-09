import sys as sys
import numpy as np
from scipy.io import FortranFile
import corrector.constants as constants
import corrector.atoms as atoms

class kdc:

    def __init__(self, filename=None):

        # KDC filename
        self.kdc_file = filename

        # Parameters of the vibronic coupling Hamiltonian
        self.n_modes  = None
        self.n_states = None
        self.n_cart   = None
        self.n_atoms  = None
        self.x0       = None
        self.q2x      = None
        self.x2q      = None
        self.atoms    = None
        
        self.order    = None
        self.e0       = None
        self.freq     = None
        self.coeff    = None
        self.dipfit   = False
                
        # Read in the model potential
        self.read_model()
        
    def read_model(self):
        """Parses a kdc data file and fills in the parameters
        of the vibronic coupling Hamiltonian"""

        f = FortranFile(self.kdc_file, 'r')

        self.n_modes  = f.read_ints(np.int32)[0]
        self.n_states = f.read_ints(np.int32)[0]
        self.n_cart   = f.read_ints(np.int32)[0]
        self.n_atoms  = f.read_ints(np.int32)[0]
        self.ngeom    = f.read_ints(np.int32)[0]
        self.order    = f.read_ints(np.int32)[0]
        self.dipfit   = f.read_ints(np.int32)[0]
        self.x0       = f.read_reals(float) * constants.bohr2ang
        atom_numbers  = f.read_ints(np.int32)
        self.atoms = [atoms.labels[atom_numbers[i]]
                      for i in range(self.n_atoms)]
        self.q2x      = f.read_reals(float).reshape((self.n_cart,
                                                     self.n_modes),
                                                    order='F')
        self.x2q      = f.read_reals(float).reshape((self.n_modes,
                                                     self.n_cart),
                                                    order='F')
        self.e0       = f.read_reals(float)
        self.freq     = f.read_reals(float)
        self.coeff    = f.read_reals(float).reshape((self.n_modes,
                                                     self.n_states,
                                                     self.n_states,
                                                     self.order),
                                                    order='F')

        return
