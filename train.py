import torch
from qucumber.nn_states import DensityMatrix
from qucumber.callbacks import MetricEvaluator
import qucumber.utils.unitaries as unitaries
import qucumber.utils.training_statistics as ts
import qucumber.utils.cplx as cplx
import qucumber.utils.data as data
import qucumber

import utils

def load_data_SV(train_data_path: str):
    meas_pattern_path = train_data_path + "measurement_pattern.txt"
    meas_label_path = train_data_path + "measurement_label.txt"
    meas_result_path = train_data_path + "measurement_result.txt"
    target_vec_path = target_state_path + "state_vector.txt"
    meas_result, target_vec, meas_label, meas_pattern = data.load_data(meas_result_path, target_vec_path, meas_label_path, meas_pattern_path)
    
    return meas_result, target_vec, meas_label, meas_pattern

def load_data_(train_data_path: str):
    meas_pattern_path = train_data_path + "measurement_pattern.txt"
    meas_label_path = train_data_path + "measurement_label.txt"
    meas_result_path = train_data_path + "measurement_result.txt"
    target_rho_re_path = target_state_path + "rho_real.txt"
    target_rho_im_path = target_state_path + "rho_imag.txt"
    meas_result, target_rho, meas_label, meas_pattern = data.load_data_DM(meas_result_path, target_rho_re_path, target_rho_im_path, meas_label_path, meas_pattern_path)
    
    return meas_result, target_rho, meas_label, meas_pattern

def ps_architecture(num_visible: int, num_hidden: int, use_gpu: bool):
    nn_state = ComplexWaveFunction(num_visible = num_visible, num_hidden = num_hidden, unitary_dict = unitaries.create_dict(), gpu = use_gpu)
    return nn_state

def ms_architecture(num_visible: int, num_hidden: int, num_aux: int, use_gpu: bool):
    nn_state = DensityMatrix(num_visible = num_visible, num_hidden = num_hidden, num_aux = n_aux_unit, unitary_dict = unitaries.create_dict(), gpu = use_gpu)
    return nn_state

def create_callback_sv(nn_state):
    metric_dict = {
        "Ideal_fidelity": ts.fidelity,
        #"Noisy_fidelity": noisy_fidelity,
        "KL_Divergence": ts.KL,
    }
    space = nn_state.generate_hilbert_space()
    callbacks = [
        MetricEvaluator(
            period,
            metric_dict,
            target = target_vec,
            bases = meas_pattern,
            verbose = True,
            space = space,
        )
    ]
    
    return callbacks

def state_vector():
    

def density_matrix():