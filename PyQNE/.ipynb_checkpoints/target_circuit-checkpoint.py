import numpy as np

import init_state as ini
import error_model as error
import quantum_gate as gate

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

def Bell(state_name, n_qubit, error_model):
    if state_name == "state_vector":
        state_vector = init_state(n_qubit, state_name)
        state_vector = state_vector @ gate.H(n_qubit,0)
        state_vector = state_vector @ gate.Cx(n_qubit,0,1)
    
    if state_name == "density_matrix":
        density_matrix = init_state(n_qubit, state_name)
        density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
        density_matrix = gate.Cx(n_qubit,0,1) @ density_matrix @ gate.Cx(n_qubit,0,1).T.conjugate()
        
        return density_matrix

def GHZ(state_name, n_qubit, error_model):
    if state_name == "state_vector":
        state_vector = init_state(n_qubit, state_name)
        state_vector = state_vector @ gate.H(n_qubit, 0)
        for i in range(n_qubit-1):
            state_vector = state_vector @ gate.Cx(n_qubit, 0, i+1)
        
        return state_vector
        
    if state_name == "density_matrix":
        density_matrix = init_state(n_qubit, state_name)
        density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
        
        for i in range(n_qubit-1):
            density_matrix = gate.Cx(n_qubit,0,i+1) @ density_matrix @ gate.Cx(n_qubit,0,i+1).T.conjugate()
            
        return density_matrix
    
"""
def random_circuit():

def ising_model()
"""