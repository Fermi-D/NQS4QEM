import numpy as np

import init_state as ini
import error_model.depolarizing_error
import error_model.unitary_error
import quantum_gate as gate

def Bell(state_name, n_qubit, error_model):
    if state_name == "state_vector":
        state_vector = ini.init_state(n_qubit, state_name)
        state_vector = state_vector @ gate.H(n_qubit, 0)
        state_vector = state_vector @ gate.Cx(n_qubit, 0, 1)
    
    if state_name == "density_matrix":
        density_matrix = ini.init_state(n_qubit, state_name)
        density_matrix = gate.H(n_qubit, 0) @ density_matrix @ gate.H(n_qubit, 0)

def GHZ(state_name, n_qubit, error_model):
    if state_name == "state_vector":
        state_vector = ini.init_state(n_qubit, state_name)
        state_vector = state_vector @ gate.H(n_qubit, 0)
        for i in range(n_qubit-1):
            state_vector = state_vector @ gate.Cx(n_qubit, 0, i+1)
        
        return state_vector
        
    if state_name == "density_matrix":
  
"""
def random_circuit():
    return 0
"""