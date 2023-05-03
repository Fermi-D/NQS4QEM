import numpy as np

import quantum_gate as gate

def depolarizing(state, n_qubit, error_rate, target_qubit_idx):
    depolarizing_term1 = (1-error_rate)*state
    depolarizing_term2 = gate.X(n_qubit, target_qubit_idx)@state@gate.X(n_qubit, target_qubit_idx) + gate.Y(n_qubit, target_qubit_idx)@state@gate.Y(n_qubit, target_qubit_idx) + gate.Z(n_qubit, target_qubit_idx)@state@gate.Z(n_qubit, target_qubit_idx)
    
    return depolarizing_term1 + (error_rate/3)*depolarizing_term2
    
def unitary(state, n_qubit, theta, target_qubit_idx):
    
    return gate.Rx(n_qubit, target_qubit_idx, theta) @ state @ gate.Rx(n_qubit, target_qubit_idx, theta).T.conjugate()