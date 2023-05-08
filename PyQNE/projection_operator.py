import numpy as np
import quantum_gate as gate

def X_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubits)
    P = gate.X(n_qubit, target_idx)
    operator_0 = (I+P) / 2
    operator_1 = (I+P) / 2
    
    return operator_0, operator_1

def Y_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubits)
    P = gate.Y(n_qubit, target_idx)
    operator_0 = (I+P) / 2
    operator_1 = (I+P) / 2
    
    return operator_0, operator_1

def Z_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubits)
    P = gate.Z(n_qubit, target_idx)
    operator_0 = (I+P) / 2
    operator_1 = (I+P) / 2
    
    return operator_0, operator_1