import numpy as np
import quantum_gate as gate

def depolarizing(state, n_qubit, error_rate, target_qubit_idx):
    I = np.eye(2)
    coff_I = (1-error_rate)*I
    
    if target_qubit_idx==0:
        mat = coff_I
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, coff_I)
        else:
            mat = np.kron(mat, I)
            
    depolarizing_term1 = mat @ state
    depolarizing_term2 = gate.X(n_qubit, target_qubit_idx)@state@gate.X(n_qubit, target_qubit_idx) + gate.Y(n_qubit, target_qubit_idx)@state@gate.Y(n_qubit, target_qubit_idx) + gate.Z(n_qubit, target_qubit_idx)@state@gate.Z(n_qubit, target_qubit_idx)
    
    return depolarizing_term1 + (error_rate/3)*depolarizing_term2
    
def unitary(state, n_qubit, theta, target_qubit_idx):
    
    return gate.Rx(n_qubit, target_qubit_idx, theta) @ state @ gate.Rx(n_qubit, target_qubit_idx, theta).T.conjugate()