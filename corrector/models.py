import numpy as np
from scipy.io import FortranFile

class kdc:

    def __init__(self, filename=None):

        # KDC filename
        self.kdc_file = filename

        # Parameters of the vibronic coupling Hamiltonian
        self.n_modes  = None
        self.n_states = None
        self.order    = None
        self.e0       = None
        self.freq     = None
        self.coeff    = None
        self.dipfit   = False

        self.x0       = None
        self.q2x      = None
        self.x2q      = None
        self.atoms    = None
        
        # Read in the model potential
        self.read_model()
        
    def read_model(self):
        """Parses a kdc data file and fills in the parameters
        of the vibronic coupling Hamiltonian"""

        f = FortranFile(self.kdc_file, 'r')

        self.n_modes  = f.read_ints(np.int32)[0]
        self.n_states = f.read_ints(np.int32)[0]
        self.ngeom    = f.read_ints(np.int32)[0]
        self.order    = f.read_ints(np.int32)[0]
        self.dipfit   = f.read_ints(np.int32)[0]
        self.e0       = f.read_reals(float)
        self.freq     = f.read_reals(float)
        self.coeff    = f.read_reals(float).reshape((self.n_modes,
                                                     self.n_states,
                                                     self.n_states,
                                                     self.order),
                                                    order='F')
        
        return
