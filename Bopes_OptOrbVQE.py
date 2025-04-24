import numpy as np
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber, ElectronicEnergy
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit_aer.primitives import Estimator
from qiskit_nature.circuit.library import HartreeFock, UCCSD

from orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbVQE

from time import perf_counter
from functools import partial
import torch
import math

###########################################################################
########################### GENERATE MATRIX ###############################
###########################################################################

#  Generate random for all columns
from scipy.linalg import qr
def generate_random_partial_unitary_matrix(rows, cols):
    """
    Membuat matriks parsial uniter dengan ukuran rows x cols.

    Parameters:
        rows (int): Jumlah baris matriks (harus >= cols).
        cols (int): Jumlah kolom matriks.

    Returns:
        np.ndarray: Matriks parsial uniter ukuran rows x cols.
    """
    if rows < cols:
        raise ValueError("Jumlah baris harus lebih besar atau sama dengan jumlah kolom untuk matriks parsial uniter.")

    # Inisialisasi matriks acak rows x cols
    random_matrix = np.random.rand(rows, cols)

    # Gram-Schmidt orthonormalization menggunakan QR decomposition
    q, _ = qr(random_matrix, mode='economic')

    q_tensor = torch.tensor(q, dtype=torch.float64)
    return q_tensor

# Molecular based matrix
def generate_permutation_matrix(n_rows: int, n_cols: int) -> torch.Tensor:
    """
    Generates a permutation matrix of size (n_rows x n_cols) to select spin orbitals 
    with the lowest unrestricted HF energy.
    
    Arguments:
    - n_rows: number of rows (total unrestricted spin orbitals, typically 2 * number of restricted orbitals)
    - n_cols: number of spin orbitals to select
    
    Column filling rules:
    - Even columns (0, 2, 4, ...): take from the alpha block (rows 0 to n_rows//2 - 1)
    - Odd columns (1, 3, 5, ...): take from the beta block (rows n_rows//2 to n_rows - 1)
    """
    half_rows = n_rows // 2
    perm_matrix = torch.zeros((n_rows, n_cols), dtype=torch.float64)

    alpha_index = 0
    beta_index = 0

    for col in range(n_cols):
        if col % 2 == 0:  # alpha orbital
            if alpha_index < half_rows:
                perm_matrix[alpha_index, col] = 1.0
                alpha_index += 1
        else:  # beta orbital
            if beta_index < half_rows:
                perm_matrix[half_rows + beta_index, col] = 1.0
                beta_index += 1

    return perm_matrix

###########################################################################
########################### INITIALIZATION ################################
###########################################################################

# region estimator, qubit_converter, charge, multiplicity
estimator = Estimator(approximation=True)
qubit_converter = QubitConverter(mapper=JordanWignerMapper())
charge = 0
multiplicity = 1
# endregion

molecule_type = "H4Square"  # Pilihan: "H2", "H4Square", "H4Linear"

basis = 'sto3g'
# basis = 'sto6g'
# basis = '321g'
# basis = '631g'
# basis = 'ccpVDZ'

# V = generate_random_partial_unitary_matrix(16, 6)
# V = generate_permutation_matrix(20, 4)
V = None

n = 1 # Number of iteration for each distance (n times for average)

# filename = f"bopes-{molecule_type}-{basis}-r{V.shape[0]}x{V.shape[1]}-{n}.csv"
filename = f"bopes-{molecule_type}-{basis}-VQE-{n}.csv"


###########################################################################
############################# SIMULATION ##################################
###########################################################################

# region molecule_type
if molecule_type == "H2":
    initial_interatomic_distance = 0.735
    distance_values = np.linspace(-0.5, 0.5, 50)  # interatomic distance range
elif molecule_type == "H4Square":
    initial_interatomic_distance = 1.23
    distance_values = np.linspace(-1, 1, 50)
elif molecule_type == "H4Linear":
    initial_interatomic_distance = 1.23
    distance_values = np.linspace(-1, 1, 50)
else:
    raise ValueError("Unrecognized molecule type. Use 'H2', 'H4Square', or 'H4Linear'.")
# endregion


with open(filename, "w") as file_out:
    for i, distance in enumerate(distance_values):
        energy_list = []
        time_list = []

        for j in range(n):
            # Perform the computation n times for averaging
            start_time = perf_counter()
            interatomic_distance = initial_interatomic_distance + distance

            print(j, interatomic_distance)

            if molecule_type == "H4Square":
                driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {interatomic_distance}; H 0 {interatomic_distance} 0; H 0 {interatomic_distance} {interatomic_distance}',
                                    charge=0,
                                    spin=0,
                                    unit=UnitsType.ANGSTROM,
                                    basis=basis)
            elif molecule_type == "H4Linear":
                driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {interatomic_distance}; H 0 0 {2 * interatomic_distance}; H 0 0 {3 * interatomic_distance}',
                                    charge=0,
                                    spin=0,
                                    unit=UnitsType.ANGSTROM,
                                    basis=basis)
            elif molecule_type == "H2":
                driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {interatomic_distance}',
                                     charge=0,
                                     spin=0,
                                     unit=UnitsType.ANGSTROM,
                                     basis=basis)
            else:
                raise ValueError("Unrecognized molecule type. Use 'H2', 'H4Square', atau 'H4Linear'.")

            q_molecule = driver.run()
            num_particles = q_molecule.get_property(ParticleNumber).num_particles

            l_bfgs_b = L_BFGS_B(maxfun=100000, maxiter=100000)

            if V is None:
                num_reduced_qubits = q_molecule.get_property(ParticleNumber).num_spin_orbitals
            else:
                num_reduced_qubits = V.shape[1]

            HF_state = HartreeFock(qubit_converter=qubit_converter,
                                   num_spin_orbitals=num_reduced_qubits,
                                   num_particles=num_particles)

            ansatz = UCCSD(qubit_converter=qubit_converter,
                           num_spin_orbitals=num_reduced_qubits,
                           num_particles=num_particles,
                           initial_state=HF_state)

            vqe_start_time = perf_counter()
            def store_intermediate_vqe_result(optorb_iteration, eval_count, parameters, mean, std):
                global vqe_start_time
                
                vqe_start_time = perf_counter()

            def get_vqe_callback(optorb_iteration: int):
                return partial(store_intermediate_vqe_result, optorb_iteration)


            orbital_rotation_start_time = perf_counter()
            def store_intermediate_orbital_rotation_result(optorb_iteration, orbital_rotation_iteration, energy):
                global orbital_rotation_start_time
                orbital_rotation_start_time = perf_counter()


            def get_orbital_rotation_callback(optorb_iteration: int):
                return partial(store_intermediate_orbital_rotation_result, optorb_iteration)


            partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                                    stopping_tolerance=10**-5,
                                                                    maxiter=100000,
                                                                    gradient_method='autograd')

            vqe_instance = VQE(ansatz=ansatz,
                               initial_point=np.zeros(ansatz.num_parameters),
                               optimizer=l_bfgs_b,
                               estimator=estimator)

            optorbvqe_instance = OptOrbVQE(molecule_driver=driver,
                                           integral_tensors=None,
                                           num_spin_orbitals=num_reduced_qubits,
                                           ground_state_solver=vqe_instance,
                                           qubit_converter=qubit_converter,
                                           estimator=estimator,
                                           initial_partial_unitary=V,
                                           partial_unitary_optimizer=partial_unitary_optimizer,
                                           maxiter=200,
                                           wavefuntion_real=True,
                                           spin_conserving=True,
                                           stopping_tolerance=10**-5,
                                           minimum_eigensolver_callback_func=get_vqe_callback,
                                           orbital_rotation_callback_func=get_orbital_rotation_callback,
                                           callback=None,
                                           spin_restricted=False,
                                           partial_unitary_random_perturbation=0.01,
                                           minimum_eigensolver_random_perturbation=0.0)

            ground_state_energy_result = optorbvqe_instance.compute_minimum_energy()
            electronic_energy = ground_state_energy_result.eigenvalue.real
            electronic_energy_property = q_molecule.get_property(ElectronicEnergy)
            nuclear_repulsion_energy = electronic_energy_property.nuclear_repulsion_energy
            total_energy = electronic_energy + nuclear_repulsion_energy

            iter_total_time = perf_counter() - start_time
            energy_list.append(total_energy)
            time_list.append(iter_total_time)
        
        # Calculates energy and time average
        avg_energy = np.mean(energy_list)
        avg_time = np.mean(time_list)
        
        # Write results into the file
        file_out.write(f"{i};{interatomic_distance};{avg_energy};{avg_time}\n")