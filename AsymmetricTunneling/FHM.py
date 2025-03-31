'''
This script performs the following calculations for a quantum system:

- Computes the expectation value of the number operator on sites after the barrier.
- Tracks the evolution of the system's energy over time.
- Generates a density map of particle distributions.
- Computes the entanglement entropy at different time steps.

''' 

from quspin.operators import hamiltonian,exp_op,quantum_operator,commutator # operators
from quspin.basis import spinful_fermion_basis_1d, base # Hilbert spaces
from quspin.tools.measurements import obs_vs_time # calculating dynamics
from matplotlib.colors import LogNorm # enable log color plot
from scipy.linalg import expm
import numpy as np # general math functions
import matplotlib.pyplot as plt
import FHM_functions as fhm
import copy


base.MAXPRINT = 1000


'''
Configure which figures to save and display during the simulation.

Set the corresponding indices to `1` to enable saving or displaying:

Index Mapping:
0 -- Densities on the sites after the barrier  
1 -- Density map  
2 -- Energy evolution    

'''

save = np.array([0, 0, 0])     # set 1 to save corresponding figures  
show_fig = np.array([1, 1, 0]) # set 1 to display corresponding plots  


#------------------------------------------------------------------------------

# boundary conditions
bc = 0                          # 0 -- open, 1 -- periodic

# system size
L = 4

# number of particles
N = 2 #L//2                      # total number of particles  
N_up = N//2 + N % 2             # number of fermions with spin up
N_down = N//2                   # number of fermions with spin down
                        
# physical parameters
J = 1.0                         # hopping strength
U = 0*J                         # interaction strength
h = 10*J                        # external potential strength

# external potential
Vex = np.zeros(L)              
Vex[int(L/2)-1] = h             # potential in middle-left
Vex[int(L/2)] = h/2            # potential in middle-right


# time evolution range
start, stop, num = 0, 50, 1000
t = np.linspace(start, stop, num = num, endpoint = True)
dt = t[1] - t[0]

# for density evolution
'''
1 - Measure n_up (density of spin-up fermions)
# psi0[6] = -1/np.sqrt(2)
2 - Measure n_down (density of spin-down fermions)
3 - Measure both n_up + n_down (total density)
'''
measure = 3     # specify the type of the measurement

# basis creation
basis = spinful_fermion_basis_1d(L, Nf = (N_up, N_down))

# set the initial system configuration 
psi0 = np.zeros(basis.Ns).reshape(-1, 1)
psi0[0] = 1     # see print(basis) for possible states

#------------------------------------------------------------------------------

# compute Hamiltonian 
H_vertical = fhm.fhm_hamiltonian(L, J, U, Vex, bc, basis) # particles face vertical side 

H_angled = fhm.fhm_hamiltonian(L, J, U, Vex[::-1], bc, basis) # particles face angled side 

# evolve the state of the system
psi_t_vertical = fhm.system_evolution(H_vertical, t, copy.deepcopy(psi0))
psi_t_angled = fhm.system_evolution(H_angled, t, copy.deepcopy(psi0))

#------------------------------------------------------------------------------

# compute evolution when particle face vertical side of the barrier
vertical = obs_vs_time(psi_t_vertical, t, dict(
    U_op = fhm.interaction_operator(basis, L, U),
    Vex_op = fhm.potential_operator(basis, L, Vex),
    t_op = fhm.tunneling_operator(basis, L, J, bc),
    **fhm.dens(L, measure, basis)))

# extract observables
dens_vertical = np.hstack([
    vertical[str(_)] for _ in range(L)
  ]).real
interaction_vertical = vertical["U_op"].real
potential_vertical = vertical["Vex_op"].real
tunneling_vertical = vertical["t_op"].real

# compute evolution when particle face angled side of the barrier
angled = obs_vs_time(psi_t_angled, t, dict(
    U_op = fhm.interaction_operator(basis, L, U),
    Vex_op = fhm.potential_operator(basis, L, Vex[::-1]),
    t_op = fhm.tunneling_operator(basis, L, J, bc),
    **fhm.dens(L, measure, basis)))

# extract observables
dens_angled = np.hstack([
    angled[str(_)] for _ in range(L)
  ]).real
interaction_angled = angled["U_op"].real
potential_angled = angled["Vex_op"].real
tunneling_angled = angled["t_op"].real

n_v = dens_vertical[:,int(L/2)+1:L].sum(axis=1) 
n_a = dens_angled[:,int(L/2)+1:L].sum(axis=1)

# plot observables

bc_str = 'PBC' if bc else 'OBC'

# densities on the sites after the barrier  
if show_fig[0]==1:
    plt.figure(1)  
    plt.plot(t, n_v, label="Vertical first")
    plt.plot(t, n_a, label="Angled first")    
    plt.legend()
    plt.xlabel("$t$ ($\hbar/J$)", fontname='Times New Roman', fontsize=20)
    plt.ylabel("density after the barrier", fontname='Times New Roman', fontsize=20)
    plt.xticks(fontname='Times New Roman', fontsize=20)
    plt.yticks(fontname='Times New Roman', fontsize=20)
    plt.title(f"U={U:.2f}, h={h}, L={L} ({bc_str})", fontname="Times New Roman", fontsize=20)
    plt.gca().set_box_aspect(1) # to make box squared
    if save[0]==1:
        filename = f"dens_{U:.2f}U_{h}h_{L}L_{stop}t_{bc_str}_sn.svg"
        plt.savefig(filename, format="svg", dpi=1200)
    plt.show()
    

# density map
if show_fig[1]==1:
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(
        dens_vertical,
        origin='lower',
        interpolation="nearest",
        norm=LogNorm(vmin=1e-3, vmax=2),
        aspect=0.4, # image aspect ratio
        extent=[0, L-1, t[0], t[-1]],
        cmap="magma"
    )
    plt.xlabel('site', fontname='Times New Roman')
    plt.xticks(np.arange(0,L), fontname='Times New Roman', fontsize=10)
    plt.yticks(fontname='Times New Roman', fontsize=10)
    plt.ylabel('$t$ ($\hbar/J$)', fontname='Times New Roman')
    plt.title('Vertical first', fontname='Times New Roman')
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(
        dens_angled,
        origin='lower',
        interpolation="nearest",
        norm=LogNorm(vmin=1e-3, vmax=2),
        aspect=0.4, # image aspect ratio
        extent=[0, L-1, t[0], t[-1]],
        cmap="magma"
    )
    plt.xlabel('site', fontname='Times New Roman')
    plt.xticks(np.arange(0,L), fontname='Times New Roman', fontsize=10)
    plt.yticks(fontname='Times New Roman', fontsize=10)
    plt.title('Angled first', fontname='Times New Roman')
    plt.colorbar()
    if save[1]==1:
        filename = f"res_trap_map_{L}L_{U:.2f}U_{h}h_{stop}t.svg"
        plt.savefig(filename, format="svg", dpi=1200)
    plt.show()


# energy evolution  
if show_fig[2]==1:
    plt.figure(3)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, tunneling_vertical, label="t")
    ax1.plot(t, interaction_vertical, label="U")
    ax1.plot(t, potential_vertical, label="Vex")
    ax1.axhline(0, linestyle='--', color='black')
    plt.legend()
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    ax1.set_title('Vertical first')
    
    ax2.plot(t, tunneling_angled, label="t")
    ax2.plot(t, interaction_angled, label="U")
    ax2.plot(t, potential_angled, label="Vex")
    ax2.axhline(0, linestyle='--', color='black')
    ax2.legend()
    ax2.set_xlabel("time")
    ax2.set_title('Angled first')
    if save[2]==1:
        filename = f"energy_{L}L_{U:.2f}U_{h}h_{stop}t.png"
        plt.savefig(filename, format="png", dpi=1200)
    plt.show()
    

