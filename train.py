import torch
from qucumber.nn_states import DensityMatrix
from qucumber.nn_states import ComplexWaveFunction
from qucumber.callbacks import MetricEvaluator
import qucumber.utils.unitaries as unitaries
import qucumber.utils.training_statistics as ts
import qucumber.utils.cplx as cplx
import qucumber.utils.data as data
import qucumber

def alpha(nn_state, space, **kwargs):
    rbm_psi = nn_state.psi(space)
    normalization = nn_state.normalization(space).sqrt_()
    alpha_ = cplx.norm(
        torch.tensor([rbm_psi[0][0], rbm_psi[1][0]], device=nn_state.device)
        / normalization
    )
    return alpha_

def beta(nn_state, space, **kwargs):
    rbm_psi = nn_state.psi(space)
    normalization = nn_state.normalization(space).sqrt_()
    beta_ = cplx.norm(
        torch.tensor([rbm_psi[0][1], rbm_psi[1][1]], device=nn_state.device)
        / normalization
    )
    return beta_

def gamma(nn_state, space, **kwargs):
    rbm_psi = nn_state.psi(space)
    normalization = nn_state.normalization(space).sqrt_()
    gamma_ = cplx.norm(
        torch.tensor([rbm_psi[0][2], rbm_psi[1][2]], device=nn_state.device)
        / normalization
    )
    return gamma_

def delta(nn_state, space, **kwargs):
    rbm_psi = nn_state.psi(space)
    normalization = nn_state.normalization(space).sqrt_()
    delta_ = cplx.norm(
        torch.tensor([rbm_psi[0][3], rbm_psi[1][3]], device=nn_state.device)
        / normalization
    )
    return delta_

def state_vector(meas_pattern_path, meas_label_path, meas_result_path, target_state_path):
    meas_pattern = data.load_data(meas_pattern_path)
    meas_label = data.load_data(meas_label_path)
    meas_result = data.load_data(meas_result_path)
    target_state = data.load_data(target_state_path)
    
    nn_state_sv = ComplexWaveFunction(num_visible = CFG.n_visible_unit, 
                                      num_hidden = CFG.n_hidden_unit, 
                                      unitary_dict = unitaries.create_dict(),
                                      gpu = True
                                     )
    
    callbacks = create_callback_sv(nn_state_sv)
    nn_state_sv.fit(data = train_samples,
                    input_bases = train_bases,
                    epochs = CFG.epochs,
                    pos_batch_size = CFG.pbs,
                    neg_batch_size = CFG.nbs,
                    lr = CFG.sv_lr,
                    k = CFG.k,
                    bases = bases,
                    callbacks = callbacks,
                    time = True,
                    #optimizer = torch.optim.Adadelta,
                    #scheduler = torch.optim.lr_scheduler.StepLR,
                    #scheduler_args = {"step_size": CFG.sv_lr_drop_epoch, "gamma": CFG.sv_lr_drop_factor},
                   )
    

def density_matrix(meas_pattern_path, meas_label_path, meas_result_path, target_state_path):
    
def max_eigen_vector(meas_pattern_path, meas_label_path, meas_result_path, target_state_path):
    