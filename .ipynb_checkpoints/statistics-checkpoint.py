import numpy as np
import utils
import sampling_VD

def bias(nn_state, ideal_state, n_copies, obs_mat):
    mat = utils.get_density_matrix(nn_state)
    pow_mat = np.linalg.matrix_power(mat, n_copy)
    vd_ev = np.trace(obs_mat@pow_mat) / np.trace(pow_mat)
    true_ev = np.trace(obs_mat @ ideal_state)
    
    return np.abs(true_ev - vd_ev)

def variance(nn_state, pauli_dict, n_samples, n_copies):
    obs_stat_dict = sampling_VD.obs_estimater(nn_state, pauli_dict, n_samples, n_copies)
    return obs_stat_dict["std_error"]**2

#def mse(nn_state, ideal_state, n_copy, obs_mat):
    
    
    