import numpy as np

def init_state(n_qubit, state_name):
    ket_0 = np.array([[1],[0]]) 
    init_state_vec = ket_0
    
    for i in range(2**(n_qubits-1)-1):
        init_state_vec = np.append(init_state_vec, np.array([[0],[0]]), axis=0)
        
    if state_name == "state vector":
        return init_state_vec # |00...0>
    
    if state_name == "density matrix":
        init_rho = init_state_vec @ init_state_vec.T.conjugate()
        return init_rho # |00...0><00...0|