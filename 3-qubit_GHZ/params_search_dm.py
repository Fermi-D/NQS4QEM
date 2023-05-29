import os
import sys
import subprocess
import random
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm.notebook import tqdm
import itertools

import torch
from qucumber.nn_states import DensityMatrix
from qucumber.nn_states import ComplexWaveFunction
from qucumber.callbacks import MetricEvaluator
import qucumber.utils.unitaries as unitaries
import qucumber.utils.training_statistics as ts
import qucumber.utils.cplx as cplx
import qucumber.utils.data as data
from qucumber.observables import ObservableBase, to_pm1
from qucumber.observables.pauli import flip_spin
import qucumber

from qulacs.gate import Pauli

import optuna

with open('./params_setting.yaml', 'r') as yml:
    params = yaml.safe_load(yml)
    
# quantum circuit parameter
n_qubit = params["circuit_info"]["n_qubit"]
each_n_shot = params["circuit_info"]["each_n_shot"]
state_name = params["circuit_info"]["state_name"]
error_model = params["circuit_info"]["error_model"]
error_rate = params["circuit_info"]["error_rate"]
# RBM architecture parameter
n_visible_unit = params["architecture_info"]["n_visible_unit"]
n_hidden_unit = params["architecture_info"]["n_hidden_unit"] 
n_aux_unit = params["architecture_info"]["n_aux_unit"]
# train parameter
lr = params["train_info"]["lr"]
pbs = params["train_info"]["positive_batch_size"]
nbs = params["train_info"]["negative_batch_size"]
n_gibbs_step = params["train_info"]["n_gibbs_step"]
period = 25
epoch = params["train_info"]["n_epoch"]
lr_drop_epoch = params["train_info"]["lr_drop_epoch"]
lr_drop_factor = params["train_info"]["lr_drop_factor"]
seed = params["train_info"]["seed"]
# sampling parameter
n_sampling = params["sampling_info"]["n_sample"]
n_copy = params["sampling_info"]["n_copy"]
# data path info
train_data_path = f"./data/{error_model}/error_prob_{100*error_rate}%/num_of_data_{each_n_shot}/"
ideal_state_path = f"./target_state/"

# settings
## warnings
warnings.simplefilter('ignore')

## seaborn layout
sns.set()
sns.set_style("white")

## seed
def seed_settings(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    qucumber.set_random_seed(seed, cpu=True, gpu=False)

seed_settings(seed=seed)

def create_callback_dm(nn_state):
    metric_dict = {
        "Fidelity": ts.fidelity,
        "KL_Divergence": ts.KL,
    }

    space = nn_state.generate_hilbert_space()
    callbacks = [
        MetricEvaluator(
            period,
            metric_dict,
            target = ideal_rho,
            bases = meas_pattern,
            verbose = True,
            space = space,
        )
    ]
    return callbacks

def objective(trial):
    # search params
    lr = trial.suggest_float("lr", 2, 20, log=True)
    k = trial.suggest_int("k", 10, 5000, log=True)
    
    # load dataset
    meas_pattern_path = train_data_path + "/measurement_pattern.txt"
    meas_label_path = train_data_path + "/measurement_label.txt"
    meas_result_path = train_data_path + "/measurement_result.txt"
    ideal_rho_re_path = ideal_state_path + "/rho_real.txt"
    ideal_rho_im_path = ideal_state_path + "/rho_imag.txt"
    meas_result, ideal_rho, meas_label, meas_pattern = data.load_data_DM(meas_result_path,
                                                                         ideal_rho_re_path,
                                                                         ideal_rho_im_path,
                                                                         meas_label_path,
                                                                         meas_pattern_path)
    
    nn_state_dm = DensityMatrix(num_visible = n_visible_unit, 
                                num_hidden = n_hidden_unit, 
                                num_aux = n_aux_unit, 
                                unitary_dict = unitaries.create_dict(),
                                gpu = False)
    
    callbacks = create_callback_dm(nn_state_dm)
    
    for step in range(100):
        nn_state_dm.fit(data = meas_result,
                        input_bases = meas_label,
                        epochs = epoch,
                        pos_batch_size = pbs,
                        neg_batch_size = nbs,
                        lr = lr,
                        k = n_gibbs_step,
                        bases = meas_pattern,
                        callbacks = callbacks,
                        time = True,
                        optimizer = torch.optim.Adadelta,
                        schexduler = torch.optim.lr_scheduler.StepLR,
                        scheduler_args = {"step_size": lr_drop_epoch, "gamma": lr_drop_factor},
                       )
        
        loss = callbacks[0]["KL_Divergence"][-1]
        trial.report(loss, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return callbacks[0]["KL_Divergence"][-1]

def main():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)  
    # save best params
    params["train_info"]["lr"] = study.best_params["lr"]
    params["train_info"]["n_gibbs_step"] = study.best_params["k"]

    with open('./best_params_setting.yaml', 'w') as yml:
        yaml.dump(params, yml, default_flow_style=False)

if __name__ == "__main__":
    main()