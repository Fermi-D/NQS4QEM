import os
import numpy as np
import pandas as pd

import qucumber.utils.data as data
import qucumber.utils.cplx as cplx

def save_density_matrix(rho, target_state_path):
    os.makedirs(target_state_path, exist_ok = True)
    np.savetxt(target_state_path+"rho_real.txt", np.real(rho))
    np.savetxt(target_state_path+"rho_imag.txt", np.imag(rho))
    
def save_state_vector(vector, target_state_path):
    os.makedirs(target_state_path, exist_ok = True)
    state_vector_df = pd.DataFrame({"Re":np.real(vector).reshape(-1), "Im":np.imag(vector).reshape(-1)})
    state_vector_df.to_csv(target_state_path + "state_vector.txt", sep="\t", header=False, index=False)
    
def load_dataset_SV(train_data_path: str, target_state_path: str):
    meas_pattern_path = train_data_path + "measurement_pattern.txt"
    meas_label_path = train_data_path + "measurement_label.txt"
    meas_result_path = train_data_path + "measurement_result.txt"
    target_vec_path = target_state_path + "state_vector.txt"
    meas_result, target_vec, meas_label, meas_pattern = data.load_data(meas_result_path, target_vec_path, meas_label_path, meas_pattern_path)
    
    return meas_result, target_vec, meas_label, meas_pattern

def load_dataset_DM(train_data_path: str, target_state_path: str):
    meas_pattern_path = train_data_path + "measurement_pattern.txt"
    meas_label_path = train_data_path + "measurement_label.txt"
    meas_result_path = train_data_path + "measurement_result.txt"
    target_rho_re_path = target_state_path + "rho_real.txt"
    target_rho_im_path = target_state_path + "rho_imag.txt"
    meas_result, target_rho, meas_label, meas_pattern = data.load_data_DM(meas_result_path, target_rho_re_path, target_rho_im_path, meas_label_path, meas_pattern_path)
    
    return meas_result, target_rho, meas_label, meas_pattern

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

def fidelity(state_1, state_2):
    F = np.trace(sqrtm(sqrtm(state_1)@state_2@sqrtm(state_1)))
    return (F.real)**2