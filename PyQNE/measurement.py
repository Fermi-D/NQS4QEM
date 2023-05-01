import numpy as np
import projection_operator as M
import target_circuit as circuit

def pauli(n_qubits, pauli_str_list):
    target_qubit_idx_list = np.arange(n_qubits)
    pauli_measurement_dict = {"X":M.X_basis, "Y":M.Y_basis, "Z":M.Z_basis}
    measured_rho_0 = circuit.Bell("density_matrix", n_qubits, )
    measured_rho_1 = circuit.Bell(n_qubits)
    measurement_result_list = []
    
    for target_qubit_idx, pauli_str in zip(target_qubit_idx_list, pauli_str_list):
        pro_ope_0, pro_ope_1 = M.