import numpy as np

def get_density_matrix(nn_state):
    space = nn_state.generate_hilbert_space()
    Z = nn_state.normalization(space)
    tensor = nn_state.rho(space, space)/Z
    tensor = cplx.conjugate(tensor)
    matrix = cplx.numpy(tensor)
    
    return matrix

def get_max_eigen_vector(matrix):
    e_val, e_vec = np.linalg.eigh(matrix)
    me_val = e_val[-1]
    me_vec = e_vec[:,-1]
    return me_vec
    
def get_state_vector(nn_state):
    space = nn_state.generate_hilbert_space()
    Z = nn_state.normalization(space)
    tensor = nn_state.psi(space, space)/Z
    vec = cplx.numpy(tensor)
    
    return vec