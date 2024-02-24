import numpy as np
import gate
import target_circuit

def X_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubit)
    P = gate.X(n_qubit, target_idx)
    operator_0 = (I + P) / 2
    operator_1 = (I - P) / 2
    
    return operator_0, operator_1

def Y_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubit)
    P = gate.Y(n_qubit, target_idx)
    operator_0 = (I + P) / 2
    operator_1 = (I - P) / 2
    
    return operator_0, operator_1

def Z_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubit)
    P = gate.Z(n_qubit, target_idx)
    operator_0 = (I + P) / 2
    operator_1 = (I - P) / 2
    
    return operator_0, operator_1

def pauli(state, n_qubit, error_model, pauli_str_list):
    target_qubit_idx_list = np.arange(n_qubit)
    pauli_meas_dict = {"X":X_basis, "Y":Y_basis, "Z":Z_basis}
    
    measurement_label_list = []
    measurement_result_list = []
    
    rho = state
    
    for target_qubit_idx, pauli_str in zip(target_qubit_idx_list, pauli_str_list):
        operator0, operator1 = pauli_meas_dict[pauli_str](n_qubit, target_qubit_idx)
        p0 = np.real(np.trace(operator0.T.conjugate() @ operator0 @ rho))
        p1 = np.real(np.trace(operator1.T.conjugate() @ operator1 @ rho))
        
        measurement_result_list.append(np.random.choice(["0", "1"], p=[p0, p1]))
        measurement_label_list.append(pauli_str)
        
        if measurement_result_list[target_qubit_idx] == "0":
            rho = (operator0 @ rho @ operator0.T.conjugate()) / np.trace(operator0.T.conjugate() @ operator0 @ rho)
        if measurement_result_list[target_qubit_idx] == "1":
            rho = (operator @ rho @ operator1.T.conjugate()) / np.trace(operator1.T.conjugate() @ operator1 @ rho)
        
    return measurement_label_list, measurement_result_list