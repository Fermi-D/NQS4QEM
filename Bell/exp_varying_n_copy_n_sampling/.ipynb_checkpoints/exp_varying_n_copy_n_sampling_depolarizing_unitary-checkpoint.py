import os
import subprocess
import random
import warnings
import numpy as np
import scipy
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

from numba import jit
from numba.experimental import jitclass

# data path info
environment = "local"
#if environment == "local":
    #with open('./params_setting.yaml', 'r') as yml:
        #params = yaml.safe_load(yml)
if environment == "colab":
    from google.colab import drive
    drive.mount("/content/drive/")
    drive_path = "/content/drive/MyDrive/NQS4QEM/Bell"
    #with open(drive_path + '/params_setting.yaml', 'r') as yml:
        #params = yaml.safe_load(yml)

#if environment == "local":
    #train_data_path = f"./data/{error_model}/error_prob_{100*error_rate}%/num_of_data_{each_n_shot}"
    #ideal_state_path = f"./target_state"
if environment == "colab":
    from google.colab import drive
    drive.mount("/content/drive/")
    drive_path = "/content/drive/MyDrive/NQS4QEM/Bell"
    #train_data_path = drive_path + f"/data/{error_model}/error_prob_{100*error_rate}%/num_of_data_{each_n_shot}"
    #ideal_state_path = drive_path + f"/target_state"


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
    qucumber.set_random_seed(seed, cpu=True, gpu=True)

seed_settings(seed=42)

class GeneralPauliDistill(ObservableBase):
    def __init__(self, pauli_dict: dict, m: int) -> None:
        self.name = "distilled_pauli"
        self.symbol = "distilled_general_pauli"
        self.pauli_dict = pauli_dict
        self.num_copy = m

    def apply(self, nn_state, samples):
        """
        This function calcualte <x1 x2 ... xm | rho^{\otimes m} O | xm x1 x2 ... xm-1> / <x1 x2 ... xm | rho^{\otimes m} | x1 x2 ... xm>
        where O acts only on the first register.
        """

        # [num_sample, num_visible_node]
        # samples = [s1, s2, s3 ... sN]
        #  where num_sample = N, and si is num_visible_node-bits
        samples = samples.to(device=nn_state.device)

        num_sample, num_visible_node = samples.shape

        # [num_sample, num_visible_node * num_copy]
        # samples_array = [[s1 sN sN-1], [s2 s1 sN], [s3 s2 s1],.. [sN sN-1 sN-2]]
        #  each row is num_copy*num_visible_node bits the above example is for num_copy=3
        samples_array = []
        for copy_index in range(self.num_copy):
            rolled_samples = torch.roll(samples, shifts=copy_index, dims=0)
            samples_array.append(rolled_samples)
        samples_array = torch.hstack(samples_array)
        assert(samples_array.shape[0] == num_sample)
        assert(samples_array.shape[1] == num_visible_node * self.num_copy)

        # roll second dim of [num_sample, num_visible_node * num_copy] by num_visible_node
        # swapped_samples_array = [[sN-1 s1 sN], [sN s2 s1], [s1 s3 s2],.. [sN-2 sN sN-1]]
        swapped_samples_array = torch.roll(samples_array, shifts = num_visible_node, dims=1)

        # pick copy of first block
        #  first_block_sample = [sN-1, sN, s1, s2, ... sN-2]
        first_block_sample = swapped_samples_array[:, :num_visible_node].clone()

        # calculate coefficient for first block [num_samples, 0:num_visible_node]
        total_prod = cplx.make_complex(torch.ones_like(samples[:,0]), torch.zeros_like(samples[:,0]))
        for index, pauli in self.pauli_dict.items():
            assert(index < num_visible_node)
            coeff = to_pm1(first_block_sample[:, index])
            if pauli == "Z":
                coeff = cplx.make_complex(coeff, torch.zeros_like(coeff))
                total_prod = cplx.elementwise_mult(coeff, total_prod)
            elif pauli == "Y":
                coeff = cplx.make_complex(torch.zeros_like(coeff), coeff)
                total_prod = cplx.elementwise_mult(coeff, total_prod)

        # flip samples for for first block [num_samples, 0:num_visible_node]
        # first_block_sample -> [OsN-1, OsN, Os1, Os2, ... OsN-2]
        #  where Osi is bit array after Pauli bit-flips
        for index, pauli in self.pauli_dict.items():
            assert(index < num_visible_node)
            if pauli in ["X", "Y"]:
                first_block_sample = flip_spin(index, first_block_sample)


        # store flipped first block
        swapped_samples_array[:, :num_visible_node] = first_block_sample

        # calculate product of coefficients
        # samples_array = [[s1 sN sN-1], [s2 s1 sN], [s3 s2 s1],.. [sN sN-1 sN-2]]
        # swapped_samples_array = [[OsN-1 s1 sN], [OsN s2 s1], [Os1 s3 s2],.. [OsN-2 sN sN-1]]
        """
        total_prod = [
            <s1 sN sN-1 | rho^{\otimes 3} | OsN-1 s1 sN> / <s1 sN sN-1 | rho^{\otimes 3} | s1 sN sN-1> ,
            <s2 s1 sN   | rho^{\otimes 3} | OsN s2 s1>   / <s2 s1 sN   | rho^{\otimes 3} | s2 s1 sN> ,
            <s3 s2 s1   | rho^{\otimes 3} | Os1 s3 s2>   / <s3 s2 s1   | rho^{\otimes 3} | s3 s2 s1> ,

        e.g.
        <s3 s2 s1   | rho^{\otimes 3} | Os1 s3 s2>   / <s3 s2 s1   | rho^{\otimes 3} | s3 s2 s1>
         = <s3 | rho | Os1> <s2 | rho | s3> < s1| rho | s2> / (<s3 | rho | s3> <s2 | rho | s2> < s1| rho | s1>)
         =  (<s3 | rho | Os1> / <s3 | rho | s3>)
          * (<s2 | rho | s3> / <s2 | rho | s2> )
          * (< s1| rho | s2> / < s1| rho | s1>)

        importance_sampling_numerator(s3, Os1)  provides <s3 | rho | Os1>
        importance_sampling_denominator(s3)     provides <s3 | rho | s3>
        """
        for copy_index in range(self.num_copy):
            st = copy_index * samples.shape[1]
            en = (copy_index+1) * samples.shape[1]
            # numerator is []
            numerator = nn_state.importance_sampling_numerator(swapped_samples_array[:, st:en], samples_array[:, st:en])
            denominator = nn_state.importance_sampling_denominator(samples_array[:, st:en])
            values = cplx.elementwise_division(numerator, denominator)
            total_prod = cplx.elementwise_mult(total_prod, values)

        value = cplx.real(total_prod)
        return value

def calculate_distilled_expectation_value(pauli_dict: dict, num_samples: int, num_copies: int):
    obs_num = GeneralPauliDistill(pauli_dict, num_copies)
    obs_div = GeneralPauliDistill({}, num_copies)
    num_stat = obs_num.statistics(nn_state_dm, num_samples=num_samples)
    div_stat = obs_div.statistics(nn_state_dm, num_samples=num_samples)

    from uncertainties import ufloat
    num = ufloat(num_stat["mean"], num_stat["std_error"])
    div = ufloat(div_stat["mean"], div_stat["std_error"])
    val = num/div
    result_dict = {"mean": val.n , "std_error": val.s, "num_samples": num_samples, "num_copies": num_copies}
    return result_dict

def get_density_matrix(nn_state):
    space = nn_state.generate_hilbert_space()
    Z = nn_state.normalization(space)
    tensor = nn_state.rho(space, space)/Z
    matrix = cplx.numpy(tensor)
    return matrix

def get_max_eigvec(matrix):
    e_val, e_vec = np.linalg.eigh(matrix)
    me_val = e_val[-1]
    me_vec = e_vec[:,-1]
    return me_vec

def get_eigvec(nn_state, obs, space, **kwargs):
    dm = get_density_matrix(nn_state)
    ev = get_max_eigvec(dm)
    ev = np.atleast_2d(ev)
    val = ev@obs@ev.T.conj()
    val = val[0,0].real
    return val

def observable_XX():
    target_list = [0, 1]
    pauli_index = [1, 1] # 1:X , 2:Y, 3:Z
    gate = Pauli(target_list, pauli_index) # = X_1 X_2
    return gate.get_matrix()

def observable_XX_ev(nn_state, **kwargs):
    obs_stat = calculate_distilled_expectation_value({0: "X", 1: "X"}, n_sampling, n_copy)
    return obs_stat["mean"]

def observable_XX_var(nn_state, **kwargs):
    obs_stat = calculate_distilled_expectation_value({0: "X", 1: "X"}, n_sampling, n_copy)
    return obs_stat["std_error"]**2

# experiment params
n_copy_list = np.arange(1, 11)[::-1]
n_sampling_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
#error_model_list = ["depolarizing", "unitary", "depolarizing&unitary"]

# number of execution
n_exec = 1000

# load_model
nn_state_dm = DensityMatrix.autoload("./model_n_pattern_shot=1000_depolarizing_unitary.pt", gpu=True)

# save result
for n_sampling in tqdm(n_sampling_list):
    each_sampling_ev_df = pd.DataFrame(columns=list(map(lambda s: "n_copy="+s, [str(i) for i in n_copy_list])))
    for n_copy in tqdm(n_copy_list):
        print(f"n_copy : {n_copy}, n_sampling : {n_sampling}")
        ev_list = []
        for i in range(n_exec):
            ev_list.append(calculate_distilled_expectation_value({0: "X", 1: "X"}, n_sampling, n_copy)["mean"])
        each_sampling_ev_df[f"n_copy={n_copy}"] = ev_list
    each_sampling_ev_df.to_csv( f"./depolarizing&unitary/ev_n_sampling={n_sampling}.csv", index = False)