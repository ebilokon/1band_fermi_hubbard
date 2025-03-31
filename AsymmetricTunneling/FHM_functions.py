

from quspin.operators import hamiltonian,exp_op,quantum_operator,commutator # operators
from quspin.basis import spinful_fermion_basis_1d # Hilbert spaces
from quspin.tools.measurements import obs_vs_time, ent_entropy # calculating dynamics
import numpy as np # general math functions
from scipy.linalg import expm

def fhm_hamiltonian(L: int, J: float, U: float, Vex: np.ndarray, bc: int, basis):
    
    '''
    Constructs the Fermi-Hubbard Model Hamiltonian for a 1D lattice.

    Arguments:
    ----------
    L (int): Number of lattice sites.
    J (float): Hopping amplitude.
    U (float): On-site interaction strength.
    Vex (np.ndarray): External potential array of shape (L,).
    bc (int): Boundary condition (0 = open boundary conditions, 
                                  1 = periodic boundary bonditions).
    basis: QuSpin basis set used for defining the Hamiltonian

    Returns:
    --------
    np.ndarray: The Hamiltonian matrix representation.
        
    Raises
    ------
    ValueError: If input dimensions are inconsistent or invalid.
    
    '''
    
    if bc not in {0, 1}:
        raise ValueError("bc must be either 0 (obc) or 1 (pbc).")

    if not isinstance(Vex, np.ndarray) or Vex.shape != (L,):
        raise ValueError(f"Vex must be a NumPy array of shape ({L},).")
        
    # Define hopping terms based on boundary conditions
    hop_right = [[-J, i, (i + 1) % L] for i in range(L if bc else L - 1)]
    hop_left = [[J, i, (i + 1) % L] for i in range(L if bc else L - 1)]    
    
    # Define on-site interaction and external potential terms
    int_list = [[U, i, i] for i in range(L)]
    
    pot_ext = [[Vex[i], i] for i in range(L)]
    
    # static Hamiltonian terms
    static = [
            ["+-|", hop_left], # up hop left
            ["-+|", hop_right], # up hop right
            ["|+-", hop_left], # down hop left
            ["|-+", hop_right], # down hop right
            ["n|n", int_list], # on-site interaction
            ["n|", pot_ext], # external potential for up
            ["|n", pot_ext], # external potential for down
            ]
    dynamic = []
    
    '''
    Optional: disable symmetry checks for performance 
    (uncomment no_checks and put **no_checks to hamiltonian() if needed)
    
    '''
    # no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    
    # construct the Hamiltonian
    H = hamiltonian(static, dynamic, basis = basis, dtype = np.float64)
    # H_matrix = H.toarray()
    
    return H #H_matrix    



def dens(L: int, measure: int, basis):
    
    '''
    Creates a matrix form of the density operator separately for each site
    for a 1D lattice.

    Arguments:
    ----------
    L (int): Number of lattice sites.
    measure (int): Specifies the type of measurement:
                1 - Measure n_up (density of spin-up fermions)
                2 - Measure n_down (density of spin-down fermions)
                3 - Measure both n_up + n_down (total density)
   basis: QuSpin basis set used for defining the Hamiltonian.

    Returns:
        dict: A dictionary where the keys are site indices, 
        and values are matrix forms of the corresponding density operators.
        
    '''
    
    if measure not in {1, 2, 3}:
        raise ValueError("measure must be 1, 2, or 3.")
        
    # initialize dictionary to store density evolution    
    dens_observables = dict()   
    
    # no symmetry checks
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    
    # loop over all sites to create density operators
    for i in range(L):  
        # creates list [[0,0],[0,1], ...,[1,# site],...,[0,L-1]]          
        identity_list = [                     
            [1 if i == _ else 0, _] for _ in range(L)
        ]
    
        if measure == 1: # measure n_up only
            dens_list = [["n|",identity_list]]  
        elif measure == 2: # measure n_down only
            dens_list = [["|n",identity_list]]  
        elif measure == 3: # measures both n_up+n_down
            dens_list = [["n|",identity_list], ["|n",identity_list]]  
            
        
        dens_observables[f"{i}"] = hamiltonian(dens_list,[],basis=basis,**no_checks)
        
    return dens_observables



def system_evolution(H: np.ndarray, t: np.ndarray, psi0: np.ndarray):
    
    '''
    Computes the time evolution of the system.
    
    Arguments:
    ----------
    H (np.ndarray): Matrix form of the Hamiltonian
    t (np.ndarray): Array of time points for time evolution.
    psi0 (np.ndarray): The initial state of the system.
    
    Returns:
    --------
    obj: generator which generates the states.
    
    Raises
    ------
    ValueError: If input dimensions are inconsistent or invalid.
    
    '''
    if not isinstance(t, np.ndarray):
        raise ValueError("t must be a NumPy array of time values.")
        
    if not isinstance(psi0, np.ndarray):
        raise ValueError("psi0 must be a NumPy array representing the initial wavefunction.")

    # use exp_op to get the evolution operator
    U = exp_op(H, a = -1j, start = t.min(), stop = t.max(), num = len(t), iterate = True)
    
    psi_t = U.dot(psi0) # get generator psi_t for time evolved state
    
    return psi_t



def interaction_operator(basis, L: int, U: float):
    
    '''
    
    Constructs the on-site interaction operator.
    
    Arguments:
    ----------
    basis: QuSpin basis set used for defining the Hamiltonian.
    L (int): Number of lattice sites.
    U (float): On-site interaction strength.
    
    Returns:
    --------
    hamiltonian: The on-site interaction operator as a QuSpin Hamiltonian object.
    
    Raises
    ------
    ValueError: If input dimensions are inconsistent or invalid.
    
    '''
    
    if not isinstance(L, int) or L <= 0:
       raise ValueError("L must be a positive integer.")
    
    # no symmetry checks
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    
    # define on-site interaction 
    int_list = [[U, i, i] for i in range(L)]
    
    static_pot = [["n|n", int_list]] # on-site interaction
    
    # construct the Hamiltonian for the interaction operator
    U_op = hamiltonian(static_pot, [], basis = basis, **no_checks)
    
    return U_op



def potential_operator(basis, L: int, Vex: np.ndarray):
    
    '''   
    Constructs the external potential operator.
    
    Arguments:
    ----------
    basis: QuSpin basis set used for defining the Hamiltonian.
    L (int): Number of lattice sites.
    Vex (np.ndarray): External potential array of shape (L,).
    
    Returns:
    --------
    hamiltonian: The external potential operator as a QuSpin Hamiltonian object.
    
    Raises
    ------
    ValueError: If input dimensions are inconsistent or invalid.
    
    '''
    
    if not isinstance(L, int) or L <= 0:
       raise ValueError("L must be a positive integer.")
       
    if not isinstance(Vex, np.ndarray) or Vex.shape != (L,):
        raise ValueError(f"Vex must be a NumPy array of shape ({L},).")
        
    # no symmetry checks
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    
    # define external potential
    pot_ext = [[Vex[i], i] for i in range(L)]
        
    static_pot = [
            ["n|", pot_ext], #external potential for up
            ["|n", pot_ext], #external potential for down
            ]
    
    # construct the Hamiltonian for the external potential operator
    Vex_op = hamiltonian(static_pot, [], basis = basis, **no_checks)
    
    return Vex_op



def tunneling_operator(basis, L: int, J: float, bc: int):
    
    '''   
    Constructs the tunneling operator.
    
    Arguments:
    ----------
    basis: QuSpin basis set used for defining the Hamiltonian.
    L (int): Number of lattice sites.
    J (float): Hopping amplitude.
    bc (int): Boundary condition (0 = open boundary conditions, 
                                  1 = periodic boundary bonditions).
    
    Returns:
    --------
    hamiltonian: The tunneling operator as a QuSpin Hamiltonian object.
    
    Raises
    ------
    ValueError: If input dimensions are inconsistent or invalid.
    
    '''
    
    if not isinstance(L, int) or L <= 0:
       raise ValueError("L must be a positive integer.")
       
    if bc not in {0, 1}:
        raise ValueError("bc must be either 0 (obc) or 1 (pbc).")
       
    # no symmetry checks
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    
    # define hopping terms based on boundary conditions
    hop_right = [[-J, i, (i + 1) % L] for i in range(L if bc else L - 1)]
    hop_left = [[J, i, (i + 1) % L] for i in range(L if bc else L - 1)]
    
    static_t = [
            ["+-|", hop_left], # up hop left
            ["-+|", hop_right], # up hop right
            ["|+-", hop_left], # down hop left
            ["|-+", hop_right], # down hop right
            ]
    
    # construct the Hamiltonian for the kinetic energy operator
    t_op = hamiltonian(static_t, [], basis = basis, **no_checks)
    
    return t_op



