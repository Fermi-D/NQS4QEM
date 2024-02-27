import numpy as np
import utils
import sampling_VD

def bias(nn_state, ideal_state, n_copies, obs_mat):
    mat = utils.get_density_matrix(nn_state)
    pow_mat = np.linalg.matrix_power(mat, n_copies)
    vd_ev = np.trace(obs_mat@pow_mat) / np.trace(pow_mat)
    true_ev = np.trace(obs_mat @ ideal_state)
    
    return np.abs(true_ev - vd_ev)

def variance(nn_state, pauli_dict, n_samples, n_copies, n_est):
    ev_list = []
    for _ in range(n_est):
        ev_list.append(sampling_VD.obs_estimater(nn_state, pauli_dict, n_samples, n_copies)["mean"])
    return np.var(ev_list)

def rmse(nn_state, ideal_state, pauli_dict, n_copies, n_samples, n_est, obs_mat):
    rmse_list = []
    for _ in range(n_est):
        sampling_vd_ev = sampling_VD.obs_estimater(nn_state, pauli_dict, n_samples, n_copies)["mean"]
        true_ev = np.trace(obs_mat @ ideal_state)
        rmse_list.append((true_ev - sampling_vd_ev)**2)
        
    return np.sqrt(np.mean(rmse_list))