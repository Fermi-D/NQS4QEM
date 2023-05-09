import numpy as np
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

def Bell(n_qubit, state_name, error_model, error_rate):
    if state_name == "state_vector":
        if error_model == "ideal":
            state_vector = init_state(n_qubit, state_name)
            state_vector = gate.H(n_qubit,0) @ state_vector
            state_vector = gate.Cx(n_qubit,0,1) @ state_vector
        
        return state_vector
    
    if state_name == "density_matrix":
        if error_model == "ideal":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
            density_matrix = gate.Cx(n_qubit,0,1) @ density_matrix @ gate.Cx(n_qubit,0,1).T.conjugate()
        
            return density_matrix
        
        if error_model == "depolarizing":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
            density_matrix = gate.Cx(n_qubit,0,1) @ density_matrix @ gate.Cx(n_qubit,0,1).T.conjugate()
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 1)
            
            return density_matrix
        
        if error_model == "unitary":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
            density_matrix = gate.Cx(n_qubit,0,1) @ density_matrix @ gate.Cx(n_qubit,0,1).T.conjugate()
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 1)
            
            return density_matrix
            
        if error_model == "depolarizing&unitary":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
            density_matrix = gate.Cx(n_qubit,0,1) @ density_matrix @ gate.Cx(n_qubit,0,1).T.conjugate()
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 1)
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 1)
            
            return density_matrix

def GHZ(n_qubit, state_name, error_model):
    if state_name == "state_vector":
        if error_model == "ideal":
            state_vector = init_state(n_qubit, state_name)
            state_vector = gate.H(n_qubit, 0) @ state_vector
            for i in range(n_qubit-1):
                state_vector = gate.Cx(n_qubit, 0, i+1) @ state_vector
            
            return state_vector
        
    if state_name == "density_matrix":
        if error_model == "ideal":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
        
            for i in range(n_qubit-1):
                density_matrix = gate.Cx(n_qubit,0,i+1) @ density_matrix @ gate.Cx(n_qubit,0,i+1).T.conjugate()
            
            return density_matrix
        
        if error_model == "depolarizing":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
            
            for i in range(n_qubit-1):
                density_matrix = gate.Cx(n_qubit,0,i+1) @ density_matrix @ gate.Cx(n_qubit,0,i+1).T.conjugate()
                density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
                density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, i+1)
            
            return density_matrix
        
        if error_model == "unitary":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
            
            for i in range(n_qubit-1):
                density_matrix = gate.Cx(n_qubit,0,i+1) @ density_matrix @ gate.Cx(n_qubit,0,i+1).T.conjugate()
                density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
                density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), i+1)
            
            return density_matrix
        
        if error_model == "depolarizing&unitary":
            density_matrix = init_state(n_qubit, state_name)
            density_matrix = gate.H(n_qubit,0) @ density_matrix @ gate.H(n_qubit,0).T.conjugate()
            density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
            density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
            
            for i in range(n_qubit-1):
                density_matrix = gate.Cx(n_qubit,0,i+1) @ density_matrix @ gate.Cx(n_qubit,0,i+1).T.conjugate()
                density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, 0)
                density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), 0)
                density_matrix = error.depolarizing(density_matrix, n_qubit, error_rate, i+1)
                density_matrix = error.unitary(density_matrix, n_qubit, np.sqrt(error_rate), i+1)
            
            return density_matrix
    
"""
def random_circuit():

def ising_model()
"""