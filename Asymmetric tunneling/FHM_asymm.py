''' 
this script calculates the expectation value of the number operator on the sites after the barrier
+ the evolution of the energy of the system 
+ density map
'''

from quspin.operators import hamiltonian,exp_op,quantum_operator,commutator # operators
from quspin.basis import spinful_fermion_basis_1d # Hilbert spaces
from quspin.tools.measurements import obs_vs_time # calculating dynamics
import numpy as np # general math functions
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # enable log color plot

'''
put 1 to positions in "save" and "show_fig": 
0 -- densities on the sites after the barrier,
1 -- density map;
2 -- energy evolution;
3 -- density at the site with h/2 barrier 
'''

save = np.array([0, 0, 0, 0])              # save figures
show_fig = np.array([1, 1, 0, 1])       # show plots

#------------------------------------------------------------------------------

# boundary conditions
bc = 0                          # 0--open, 1--periodic
# physical parameters
J = 1.0                         # hopping strength
U = 0.5*J                         # interaction strength
h = 10*J                        # external potential strength
L = 6                           # system size
Vex = np.zeros(L)               # external potential array
Vex[int(L/2)-1] = h             # create potential in the middle of the lattice
Vex[int(L/2)] = h/2
N = 2 #L//2                      # number of particles
N_up = N//2 + N % 2             # number of fermions with spin up
N_down = N//2                   # number of fermions with spin down

# range in time to evolve the system
start, stop, num = 0, 50, 1001
t = np.linspace(start, stop, num = num, endpoint = True)

#------------------------------------------------------------------------------

# basis creation
basis = spinful_fermion_basis_1d(L, Nf = (N_up, N_down))

# site-coupling lists definition without Vex
if bc == 0:
    hop_right = [[-J,i,i+1] for i in range(L-1)] # hopping to the right OBC
    hop_left = [[J,i,i+1] for i in range(L-1)] # hopping to the left OBC
else:
    hop_right = [[-J,i,(i+1)%L] for i in range(L)] # hopping to the right PBC
    hop_left = [[J,i,(i+1)%L] for i in range(L)] # hopping to the left PBC

int_list = [[U,i,i] for i in range(L)] # on-site interaction


dens_observables = dict()   # dictionary to store densities evolution on each site
no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

for i in range(L):    
    identity_list = [      # creates list [[0,0],[0,1], ...,[1,# site],...,[0,L-1]]                         
        [1 if i == _ else 0, _] for _ in range(L)
    ]
    
    dens_list = [
        ["n|",identity_list], ["|n",identity_list]  # measures both n_up+n_down
        # ["n|",identity_list]  # measures n_up
        # ["|n",identity_list]  # measures n_down
    ]
    
    dens_observables[f"{i}"] = hamiltonian(dens_list,[],basis=basis,**no_checks)

#------------------------------------------------------------------------------
'''
this functioin takes the external potential list and calculates the evolution 
of densities on each site as well as the evolution of the energy of the system  
'''
def get_density(Vex):

    # build Hamiltonian with Vex
    pot_ext = [[Vex[i],i] for i in range(L)] # external potential
    
    static = [
            ["+-|", hop_left], # up hop left
            ["-+|", hop_right], # up hop right
            ["|+-", hop_left], # down hop left
            ["|-+", hop_right], # down hop right
            ["n|n", int_list], # onsite interaction
            ["n|", pot_ext], #external potential for up
            ["|n", pot_ext], #external potential for down
            ]
    dynamic = []
    #no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H = hamiltonian(static, dynamic, basis = basis, dtype = np.float64)
    H_matrix = H.toarray().round()
    
#------------------------------------------------------------------------------
    
    # ground-state calculation
    E, V = H.eigh()
    # E_GS, V_GS = H.eigsh(k=1,which='SA',maxiter=1E9) # k=1 --> only GS

    # set the initial system configuration 
    psi0 = np.zeros(len(E)).reshape(-1, 1)
    psi0[0] = 1
    # psi0[1] = 1/np.sqrt(2)
    # psi0[6] = 1/np.sqrt(2)
    
    
    
#------------------------------------------------------------------------------
    
    # evolution
    # use exp_op to get the evolution operator
    U = exp_op(H, a = -1j, start = t.min(), stop = t.max(), num = len(t), iterate = True)
    psi_t = U.dot(psi0) # get generator psi_t for time evolved state
    
    # potential energy calculation
    # creation of the U_op (the interaction between particles)
    static_pot = [
            ["n|n", int_list], # onsite interaction
            ]
    U_op = hamiltonian(static_pot, [], basis = basis, **no_checks)
    
    
    # creation of the Vex_op (the potential barrier)
    static_pot = [
            ["n|", pot_ext], #external potential for up
            ["|n", pot_ext], #external potential for down
            ]
    Vex_op = hamiltonian(static_pot, [], basis = basis, **no_checks)
    
    
    # kinetic energy
    # creation of the t_op
    static_t = [
            ["+-|", hop_left], # up hop left
            ["-+|", hop_right], # up hop right
            ["|+-", hop_left], # down hop left
            ["|-+", hop_right], # down hop right
            ]
    t_op = hamiltonian(static_t, [], basis = basis, **no_checks)
    
    # use obs_vs_time to evaluate the dynamics
    obs_t = obs_vs_time(psi_t, t, dict(t_op = t_op, U_op = U_op, Vex_op = Vex_op, **dens_observables))     
    return obs_t, H_matrix
    
#------------------------------------------------------------------------------
'''
results + plots
vertical - particles face vertical side of the barrier first 
angled - particles face angled side of the barrier first
'''

vertical, H_v = get_density(Vex)
densities_vert = np.hstack([
    vertical[str(_)] for _ in range(L)
 ]).real

angled, H_a = get_density(Vex[::-1])
densities_angl = np.hstack([
    angled[str(_)] for _ in range(L)
 ]).real


if bc == 0:
    bc_str = 'OBC'
else: bc_str = 'PBC'


# figure 1 (densities at the sites after the barrier)
if show_fig[0]==1:
    plt.figure(1)
    plt.plot(t, np.sum(densities_vert[:,int(L/2)+1:L],axis=1), label="Vertical first")
    plt.plot(t, np.sum(densities_angl[:,int(L/2)+1:L],axis=1), label="Angled first")    
    plt.legend()
    plt.xlabel("$t$ ($J/\hbar$)", fontname='Times New Roman', fontsize=20)
    plt.ylabel("density after the barrier", fontname='Times New Roman', fontsize=20)
    plt.xticks(fontname='Times New Roman', fontsize=20)
    plt.yticks(fontname='Times New Roman', fontsize=20)
    plt.title("U=%0.2f*J," %U + " h=%d" %h +' L=%d' %L + '(%s)' %bc_str)
    plt.gca().set_box_aspect(1) # to make box squared
    if save[0]==1:
        #plt.savefig('FHM_density_%0.2fU_' %U + '%dh_' %h + '%dL_' %L + '%dt_' %stop + '%s.png' %bc_str, format='png', dpi=1200)
        plt.savefig('res_doub_%0.2fU_' %U + '%dh_' %h + '%dL_' %L + '%dt_' %stop + '%s.svg' %bc_str, format='svg', dpi=1200)
    plt.show()

# figure 2 (density map)
if show_fig[1]==1:
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(
        densities_vert,
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
    plt.ylabel('$t$ ($J/\hbar$)', fontname='Times New Roman')
    plt.title('Vertical first', fontname='Times New Roman')
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(
        densities_angl,
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
    #plt.ylabel('time $t$ (a.u.)')
    plt.title('Angled first', fontname='Times New Roman')
    plt.colorbar()
    if save[1]==1:
        #plt.savefig('FHM_diagram3_%dL_' %L + '%0.2fU_' %U + '%dh_' %h + '%dt_' %stop + '.png', format='png', dpi=1200)
        plt.savefig('res_trap_map_%dL_' %L + '%0.2fU_' %U + '%dh_' %h + '%dt_' %stop + '.svg', format='svg', dpi=1200)
    plt.show()

# figure 3 (energy evolution)
if show_fig[2]==1:
    plt.figure(3)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, vertical["t_op"].real, label="t")
    ax1.plot(t, vertical["U_op"].real, label="U")
    ax1.plot(t, vertical["Vex_op"].real, label="Vex")
    ax1.axhline(0, linestyle='--', color='black')
    plt.legend()
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    ax1.set_title('Vertical first')
    
    ax2.plot(t, angled["t_op"].real, label="t")
    ax2.plot(t, angled["U_op"].real, label="U")
    ax2.plot(t, angled["Vex_op"].real, label="Vex")
    ax2.axhline(0, linestyle='--', color='black')
    ax2.legend()
    ax2.set_xlabel("time")
    #plt.ylabel("kinetic energy")
    ax2.set_title('Angled first')
    if save[2]==1:
        plt.savefig('FHM_energy3_%dL_' %L + '%0.2fU_' %U + '%dh_' %h + '%dt_' %stop + '.png', format='png', dpi=1200)
    plt.show()


# figure 4 (density at the site with Vex = h/2)
if show_fig[3]==1:  
    plt.figure(4)
    plt.plot(t, densities_vert[:,int(L/2)], label="Vertical first")
    plt.plot(t, densities_angl[:,int(L/2-1)], label="Angled first")  
    # plt.plot(t, vertical["U_op"].real, label="U vertical") 
    # plt.plot(t, angled["U_op"].real, label="U angled")  
    plt.legend()
    plt.xlabel("$t$ ($J/\hbar$)", fontname='Times New Roman', fontsize=20)
    plt.ylabel("density h/2", fontname='Times New Roman', fontsize=20)
    plt.xticks(fontname='Times New Roman', fontsize=20)
    plt.yticks(fontname='Times New Roman', fontsize=20)
    plt.title("U=%0.2f*J," %U + " h=%d" %h +' L=%d' %L + '(%s)' %bc_str)
    #plt.xlim((4,6))
    if save[3]==1:
        #plt.savefig('FHM_density_h2_1_%0.2fU_' %U + '%dh_' %h + '%dL_' %L + '%dt_' %stop + '%s.svg' %bc_str, format='svg', dpi=1200)
        plt.savefig('density_h2_res_3p_%0.2fU_' %U + '%dh_' %h + '%dL_' %L + '%dt_' %stop + '%s.png' %bc_str, format='png', dpi=1200)
    plt.show()

print("U = ", U)
print("density on h/2 site (vertical): ", np.max(densities_vert[:,int(L/2)]))
print("density on h/2 site (angled): ", np.max(densities_angl[:,int(L/2-1)]))



