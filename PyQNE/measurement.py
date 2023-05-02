import numpy as np
import projection_operator as meas_operator
import target_circuit as circuit

def pauli(n_qubit, pauli_str_list):
    target_qubit_idx_list = np.arange(n_qubits)
    pauli_meas_dict = {"X":meas_operator.X_basis, "Y":meas_operator.Y_basis, "Z":meas_operator.Z_basis}
    measured_rho_0 = circuit.Bell("density_matrix", n_qubits, )
    measured_rho_1 = circuit.Bell(n_qubits)
    measurement_result_list = []
    
    for target_qubit_idx, pauli_str in zip(target_qubit_idx_list, pauli_str_list):
        operator0, operator1 = pauli_meas_dict[pauli_str](n_qubit, target_qubit_idx)
        p0 = np.trace(oprtr_0 @ measurement_rho_0)
        p1 = np.trace(oprtr_1 @ measurement_rho_1)
        np.random.choice(["0","1"], p=[p_0,p_1])