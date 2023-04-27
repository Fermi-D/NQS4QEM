import numpy as np
import qulacs

import init_state as ini
import error_model.depolarizing_error
import error_model.unitary_error
import quantum_gate as qg

def GHZ(state_name, n_qubit, error_model):
    try:
        if state_name == "state_vector":
            init_vector = ini.init_state(n_qubit, state_name)
            state_vector = init_vector @ qg.H(n_qubit, 0)
            state_vector = state_vector @ qg.Cx(n_qubit, 0, 1)
            
        
        if state_name == "density_matrix":
    
    except:
        print("state name is state_vector or density_matrix")
        
  
"""
def random_circuit():
    return 0
"""