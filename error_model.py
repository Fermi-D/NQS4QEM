import numpy as np
import gate

def global_depolarizing(state, n_qubit, error_rate):
    return (1-error_rate)*state + error_rate*np.trace(state)*np.eye(2**n_qubit)/(2**n_qubit)
    
def unitary(state, n_qubit, theta, target_qubit_idx):
    return gate.Rx(n_qubit, target_qubit_idx, theta) @ state @ gate.Rx(n_qubit, target_qubit_idx, theta).T.conjugate()