import numpy as np
import gate
import error_model

'''
import importlib
importlib.reload(gate)
importlib.reload(error_model)
'''

def init_state(n_qubit, state_class):
    ket_0 = np.array([[1],[0]]) 
    init_state = ket_0
    
    for i in range(2**(n_qubit-1)-1):
        init_state = np.append(init_state, np.array([[0],[0]]), axis=0) # |00...0>
    
    if state_class == "density_matrix":
        init_state_vec = init_state
        init_state = init_state_vec @ init_state_vec.T.conjugate() # |00...0><00...0|
    
    return init_state

def GHZ(n_qubit, state_class, error_model, error_rate):
    if state_class == "state_vector":
        if error_model == "ideal":
            state = init_state(n_qubit, state_name)
            state = gate.H(n_qubit, 0) @ state
            
            for i in range(n_qubit-1):
                state = gate.CX(n_qubit, 0, i+1) @ state
        
    if state_class == "density_matrix":
        if error_model == "ideal":
            state = init_state(n_qubit, state_class)
            state = gate.H(n_qubit,0) @ state @ gate.H(n_qubit,0).T.conjugate()
            
            for i in range(n_qubit-1):
                state = gate.CX(n_qubit,0,i+1) @ state @ gate.CX(n_qubit,0,i+1).T.conjugate()
        
        if error_model == "depolarizing":
            state = init_state(n_qubit, state_class)
            state = gate.H(n_qubit,0) @ state @ gate.H(n_qubit,0).T.conjugate()
            
            for i in range(n_qubit-1):
                state = gate.CX(n_qubit,0,i+1) @ state @ gate.CX(n_qubit,0,i+1).T.conjugate()
                
            state = error_model.global_depolarizing(state, n_qubit, error_rate)
        
        if error_model == "unitary":
            state = init_state(n_qubit, state_class)
            state = gate.H(n_qubit,0) @ state @ gate.H(n_qubit,0).T.conjugate()
            
            for i in range(n_qubit-1):
                state = gate.CX(n_qubit,0,i+1) @ state @ gate.CX(n_qubit,0,i+1).T.conjugate()
                
            for i in range(n_qubit):
                state = error_model.unitary(state, n_qubit, np.arcsin(np.sqrt(error_rate)), target_qubit_idx)
            
        if error_model == "depolarizing&unitary":
            state = init_state(n_qubit, state_class)
            state = gate.H(n_qubit,0) @ state @ gate.H(n_qubit,0).T.conjugate()
            
            for i in range(n_qubit-1):
                state = gate.CX(n_qubit,0,i+1) @ state @ gate.CX(n_qubit,0,i+1).T.conjugate()
                
            for i in range(n_qubit):
                state = error_model.unitary(state, n_qubit, np.arcsin(np.sqrt(error_rate)), target_qubit_idx)
            
            state = error_model.global_depolarizing(state, n_qubit, error_rate)
            
    return state

def Random_Unitary(n_qubit, state_name, error_model, error_rate, seed):
    from scipy.stats import unitary_group
    U = unitary_group.rvs(dim = 2**n_qubit, random_state = seed)
    
    if state_name == "state_vector":
        if error_model == "ideal":
            state = init_state(n_qubit, state_name)
            state = U @ state
            
    if state_name == "density_matrix":
        if error_model == "ideal":
            state = init_state(n_qubit, state_name)
            state = U @ state @ U.T.conjugate()
        
        if error_model == "depolarizing":
            state = init_state(n_qubit, state_name)
            state = U @ state @ U.T.conjugate()
            state = error_model.global_depolarizing(state, n_qubit, error_rate)
        
        if error_model == "unitary":
            state = init_state(n_qubit, state_name)
            state = U @ state @ U.T.conjugate()
            for i in range(n_qubit):
                state = error_model.unitary(state, n_qubit, np.arcsin(np.sqrt(error_rate)), i)
            
        if error_model == "depolarizing&unitary":
            state = init_state(n_qubit, state_name)
            state = U @ state @ U.T.conjugate()
            state = error_model.global_depolarizing(state, n_qubit, error_rate)
            for i in range(n_qubit):
                state = error_model.unitary(state, n_qubit, np.arcsin(np.sqrt(error_rate)), i)
    
    return state