import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm.notebook import tqdm
import itertools

with open('./params_setting.yaml', 'r') as yml:
    params = yaml.safe_load(yml)
    
# quantum circuit parameter
n_qubit = params["circuit_info"]["n_qubit"]
n_data = params["circuit_info"]["n_data"]
each_n_shot = int(n_data / 3**n_qubit)
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
period = 1
epochs = params["train_info"]["n_epoch"]
lr_drop_epoch = params["train_info"]["lr_drop_epoch"]
lr_drop_factor = params["train_info"]["lr_drop_factor"]
seed = params["train_info"]["seed"]
# sampling parameter
n_sampling = params["sampling_info"]["n_sample"]
n_copy = params["sampling_info"]["n_copy"]
# data path info
train_data_path = f"./data/{error_model}/error_prob_{100*error_rate}%/num_of_data_{n_data}/"
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
    #torch.manual_seed(seed)
    #qucumber.set_random_seed(seed, cpu=True, gpu=False)

seed_settings(seed=seed)

# 1-qubit gate
## pauli X
def X(n_qubit, target_qubit_idx):
    I = np.eye(2)
    local_X = np.array([[0,1],[1,0]])
    if target_qubit_idx==0:
        mat = local_X
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_X)
        else:
            mat = np.kron(mat, I)
            
    return mat

## pauli Y
def Y(n_qubit, target_qubit_idx):
    I = np.eye(2)
    local_Y = np.array([[0,-1j], [1j,0]])
    if target_qubit_idx==0:
        mat = local_Y
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_Y)
        else:
            mat = np.kron(mat, I)
            
    return mat

## pauli Z
def Z(n_qubit, target_qubit_idx):
    I = np.eye(2)
    local_Z = np.array([[1,0], [0,-1]])
    if target_qubit_idx==0:
        mat = local_Z
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_Z)
        else:
            mat = np.kron(mat, I)
            
    return mat

## Hadamard gate
def H(n_qubit, target_qubit_idx):
    I = np.eye(2)
    local_H = np.array([[1,1], [1,-1]]) / np.sqrt(2)
    if target_qubit_idx==0:
        mat = local_H
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_H)
        else:
            mat = np.kron(mat, I)
            
    return mat

## S gate
def S(n_qubit, target_qubit_idx):
    I = np.eye(2)
    local_S = np.array([[1,0], [0,1j]])
    if target_qubit_idx==0:
        mat = local_S
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_S)
        else:
            mat = np.kron(mat, I)
            
    return mat

## T gate
def T(n_qubit, target_qubit_idx):
    I = np.eye(2)
    local_T = np.array([[1,0], [0,-np.exp(1j*np.pi/4)]])
    if target_qubit_idx==0:
        mat = local_T
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_T)
        else:
            mat = np.kron(mat, I)
            
    return mat

## Rx gate
def Rx(n_qubit, target_qubit_idx, theta):
    I = np.eye(2)
    local_Rx = np.array([[np.cos(theta/2),-1j*np.sin(theta/2)], [-1j*np.sin(theta/2),np.cos(theta/2)]])
    if target_qubit_idx==0:
        mat = local_Rx
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_Rx)
        else:
            mat = np.kron(mat, I)
            
    return mat

## Ry gate
def Ry(n_qubit, target_qubit_idx, theta):
    I = np.eye(2)
    local_Ry = np.array([[np.cos(theta/2),-np.sin(theta/2)], [-np.sin(theta/2),np.cos(theta/2)]])
    if target_qubit_idx==0:
        mat = local_Ry
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_Ry)
        else:
            mat = np.kron(mat, I)
            
    return mat

## Rz gate
def Rz(n_qubit, target_qubit_idx, theta):
    I = np.eye(2)
    local_Rz = np.array([[np.exp(-1j*theta/2),0], [0,np.cos(1j*theta/2)]])
    if target_qubit_idx==0:
        mat = local_Rz
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, local_Rz)
        else:
            mat = np.kron(mat, I)
            
    return mat

# 2-qubit gate
## CX gate
def CX(n_qubit, control_qubit_idx, target_qubit_idx):
    I = np.eye(2)
    ket_0 = np.array([[1],[0]]) 
    ket_1 = np.array([[0],[1]])
    
    mat_00 = ket_0 @ ket_0.T.conjugate() ### |0><0|
    mat_11 = ket_1 @ ket_1.T.conjugate() ### |1><1|
    
    eye_tensor = I
    
    for i in range(n_qubit-2):
        eye_tensor = np.kron(eye_tensor, I)
    
    if control_qubit_idx < target_qubit_idx:
        ### |0><0|　\otimes I \otimes ... \otimes I
        cx_mat_term1 = np.kron(mat_00, eye_tensor)
        ### |1><1| \otimes I ... \otimes X ...
        cx_mat_term2 = np.kron(mat_11, X(n_qubit-1, target_qubit_idx-1))
        
    if control_qubit_idx > target_qubit_idx:
        ### I \otimes ... \otimes I \otimes |0><0|
        cx_mat_term1 = np.kron(eye_tensor, mat_00)
        ### ... \otimes X ... \otimes |1><1|
        cx_mat_term2 = np.kron(X(n_qubit-1, target_qubit_idx), mat_11)
    
    mat = cx_mat_term1 + cx_mat_term2
    
    return mat

## CY gate
def CY(n_qubit, control_qubit_idx, target_qubit_idx):
    I = np.eye(2)
    ket_0 = np.array([[1],[0]]) 
    ket_1 = np.array([[0],[1]])
    
    mat_00 = ket_0 @ ket_0.T.conjugate() ### |0><0|
    mat_11 = ket_1 @ ket_1.T.conjugate() ### |1><1|
    
    eye_tensor = I
    
    for i in range(n_qubit-2):
        eye_tensor = np.kron(eye_tensor, I)
    
    if control_qubit_idx < target_qubit_idx:
        ### |0><0|　\otimes I \otimes ... \otimes I
        cx_mat_term1 = np.kron(mat_00, eye_tensor)
        ### |1><1| \otimes ... \otimes Y ...
        cx_mat_term2 = np.kron(mat_11, Y(n_qubit-1, target_qubit_idx-1))
        
    if control_qubit_idx > target_qubit_idx:
        ### I \otimes ... \otimes I \otimes |0><0|
        cx_mat_term1 = np.kron(eye_tensor, mat_00)
        ### ... \otimes Y ... \otimes |1><1|
        cx_mat_term2 = np.kron(Y(n_qubit-1, target_qubit_idx), mat_11)
    
    mat = cx_mat_term1 + cx_mat_term2
    
    return mat

## CZ gate
def CZ(n_qubit, control_qubit_idx, target_qubit_idx):
    I = np.eye(2)
    ket_0 = np.array([[1],[0]]) 
    ket_1 = np.array([[0],[1]])
    
    mat_00 = ket_0 @ ket_0.T.conjugate() ### |0><0|
    mat_11 = ket_1 @ ket_1.T.conjugate() ### |1><1|
    
    eye_tensor = I
    
    for i in range(n_qubit-2):
        eye_tensor = np.kron(eye_tensor, I)
    
    if control_qubit_idx < target_qubit_idx:
        ### |0><0|　\otimes I \otimes ... \otimes I
        cx_mat_term1 = np.kron(mat_00, eye_tensor)
        ### |1><1| \otimes I ... \otimes Z ...
        cx_mat_term2 = np.kron(mat_11, Z(n_qubit-1, target_qubit_idx-1))
        
    if control_qubit_idx > target_qubit_idx:
        ### I \otimes ... \otimes I \otimes |0><0|
        cx_mat_term1 = np.kron(eye_tensor, mat_00)
        ### ... \otimes Z ... \otimes |1><1|
        cx_mat_term2 = np.kron(Z(n_qubit-1, target_qubit_idx), mat_11)
    
    mat = cx_mat_term1 + cx_mat_term2
    
    return mat

## CH gate
def CH(n_qubit, control_qubit_idx, target_qubit_idx):
    I = np.eye(2)
    ket_0 = np.array([[1],[0]]) 
    ket_1 = np.array([[0],[1]])
    
    mat_00 = ket_0 @ ket_0.T.conjugate() ### |0><0|
    mat_11 = ket_1 @ ket_1.T.conjugate() ### |1><1|
    
    eye_tensor = I
    
    for i in range(n_qubit-2):
        eye_tensor = np.kron(eye_tensor, I)
    
    if control_qubit_idx < target_qubit_idx:
        ### |0><0|　\otimes I \otimes ... \otimes I
        cx_mat_term1 = np.kron(mat_00, eye_tensor)
        ### |1><1| \otimes I ... \otimes H ...
        cx_mat_term2 = np.kron(mat_11, H(n_qubit-1, target_qubit_idx-1))
        
    if control_qubit_idx > target_qubit_idx:
        ### I \otimes ... \otimes I \otimes |0><0|
        cx_mat_term1 = np.kron(eye_tensor, mat_00)
        ### ... \otimes H ... \otimes |1><1|
        cx_mat_term2 = np.kron(H(n_qubit-1, target_qubit_idx), mat_11)
    
    mat = cx_mat_term1 + cx_mat_term2
    
    return mat

## CRx gate
def CRx(n_qubit, control_qubit_idx, target_qubit_idx, theta):
    I = np.eye(2)
    ket_0 = np.array([[1],[0]]) 
    ket_1 = np.array([[0],[1]])
    
    mat_00 = ket_0 @ ket_0.T.conjugate() ### |0><0|
    mat_11 = ket_1 @ ket_1.T.conjugate() ### |1><1|
    
    eye_tensor = I
    
    for i in range(n_qubit-2):
        eye_tensor = np.kron(eye_tensor, I)
    
    if control_qubit_idx < target_qubit_idx:
        ### |0><0|　\otimes I \otimes ... \otimes I
        cx_mat_term1 = np.kron(mat_00, eye_tensor)
        ### |1><1| \otimes I ... \otimes Rx ...
        cx_mat_term2 = np.kron(mat_11, Rx(n_qubit-1, target_qubit_idx-1, theta))
        
    if control_qubit_idx > target_qubit_idx:
        ### I \otimes ... \otimes I \otimes |0><0|
        cx_mat_term1 = np.kron(eye_tensor, mat_00)
        ### ... \otimes Rx ... \otimes |1><1|
        cx_mat_term2 = np.kron(Rx(n_qubit-1, target_qubit_idx, theta), mat_11)
    
    mat = cx_mat_term1 + cx_mat_term2
    
    return mat

## CRy gate
def CRy(n_qubit, control_qubit_idx, target_qubit_idx, theta):
    I = np.eye(2)
    ket_0 = np.array([[1],[0]]) 
    ket_1 = np.array([[0],[1]])
    
    mat_00 = ket_0 @ ket_0.T.conjugate() ### |0><0|
    mat_11 = ket_1 @ ket_1.T.conjugate() ### |1><1|
    
    eye_tensor = I
    
    for i in range(n_qubit-2):
        eye_tensor = np.kron(eye_tensor, I)
    
    if control_qubit_idx < target_qubit_idx:
        ### |0><0|　\otimes I \otimes ... \otimes I
        cx_mat_term1 = np.kron(mat_00, eye_tensor)
        ### |1><1| \otimes I ... \otimes Ry ...
        cx_mat_term2 = np.kron(mat_11, Ry(n_qubit-1, target_qubit_idx-1, theta))
        
    if control_qubit_idx > target_qubit_idx:
        ### I \otimes ... \otimes I \otimes |0><0|
        cx_mat_term1 = np.kron(eye_tensor, mat_00)
        ### ... \otimes Ry ... \otimes |1><1|
        cx_mat_term2 = np.kron(Ry(n_qubit-1, target_qubit_idx, theta), mat_11)
    
    mat = cx_mat_term1 + cx_mat_term2
    
    return mat

## CRz gate
def CRz(n_qubit, control_qubit_idx, target_qubit_idx, theta):
    I = np.eye(2)
    ket_0 = np.array([[1],[0]]) 
    ket_1 = np.array([[0],[1]])
    
    mat_00 = ket_0 @ ket_0.T.conjugate() ### |0><0|
    mat_11 = ket_1 @ ket_1.T.conjugate() ### |1><1|
    
    eye_tensor = I
    
    for i in range(n_qubit-2):
        eye_tensor = np.kron(eye_tensor, I)
    
    if control_qubit_idx < target_qubit_idx:
        ### |0><0|　\otimes I \otimes ... \otimes I
        cx_mat_term1 = np.kron(mat_00, eye_tensor)
        ### |1><1| \otimes I ... \otimes Rz ...
        cx_mat_term2 = np.kron(mat_11, Rz(n_qubit-1, target_qubit_idx-1, theta))
        
    if control_qubit_idx > target_qubit_idx:
        ### I \otimes ... \otimes I \otimes |0><0|
        cx_mat_term1 = np.kron(eye_tensor, mat_00)
        ### ... \otimes Rz ... \otimes |1><1|
        cx_mat_term2 = np.kron(Rz(n_qubit-1, target_qubit_idx, theta), mat_11)
    
    mat = cx_mat_term1 + cx_mat_term2
    
    return mat

## SWAP gate
def SWAP(n_qubit, qubit_idx_1, qubit_idx_2):
    mat = CX(n_qubit, qubit_idx_1, qubit_idx_2) @ CX(n_qubit, qubit_idx_2, qubit_idx_1)
    mat = mat @ CX(n_qubit, qubit_idx_1, qubit_idx_2)
    
    return mat

# 3-qubit gate
## toffoli gate
"""
def toffoli(n_qubit, control_qubit_idx_1, control_qubit_idx_2, target_qubit_idx):
"""

def depolarizing(state, n_qubit, error_rate, target_qubit_idx):
    I = np.eye(2)
    coff_I = (1-error_rate)*I
    
    if target_qubit_idx==0:
        mat = coff_I
    else:
        mat = I
    for i in range(n_qubit-1):
        if i+1==target_qubit_idx:
            mat = np.kron(mat, coff_I)
        else:
            mat = np.kron(mat, I)
            
    depolarizing_term1 = mat @ state
    depolarizing_term2 = X(n_qubit, target_qubit_idx)@state@X(n_qubit, target_qubit_idx) + Y(n_qubit, target_qubit_idx)@state@Y(n_qubit, target_qubit_idx) + Z(n_qubit, target_qubit_idx)@state@Z(n_qubit, target_qubit_idx)
    
    return depolarizing_term1 + (error_rate/3)*depolarizing_term2
    
def unitary(state, n_qubit, theta, target_qubit_idx):
    
    return Rx(n_qubit, target_qubit_idx, theta) @ state @ Rx(n_qubit, target_qubit_idx, theta).T.conjugate()

def init_state(n_qubit, state_name):
    ket_0 = np.array([[1],[0]]) 
    init_state = ket_0
    
    for i in range(2**(n_qubit-1)-1):
        init_state = np.append(init_state, np.array([[0],[0]]), axis=0) # |00...0>
    
    if state_name == "density_matrix":
        init_state_vec = init_state
        init_state = init_state_vec @ init_state_vec.T.conjugate() # |00...0><00...0|
    
    return init_state

def Bell(n_qubit, state_name, error_model, error_rate):
    if state_name == "state_vector":
        if error_model == "ideal":
            state = init_state(n_qubit, state_name)
            state = H(n_qubit,0) @ state
            state = CX(n_qubit,0,1) @ state
    
    if state_name == "density_matrix":
        if error_model == "ideal":
            state = init_state(n_qubit, state_name)
            state = H(n_qubit,0) @ state @ H(n_qubit,0).T.conjugate()
            state = CX(n_qubit,0,1) @ state @ CX(n_qubit,0,1).T.conjugate()
        
        if error_model == "depolarizing":
            state = init_state(n_qubit, state_name)
            state = H(n_qubit,0) @ state @ H(n_qubit,0).T.conjugate()
            state = depolarizing(state, n_qubit, error_rate, 0)
            state = CX(n_qubit,0,1) @ state @ CX(n_qubit,0,1).T.conjugate()
            state = depolarizing(state, n_qubit, error_rate, 0)
            state = depolarizing(state, n_qubit, error_rate, 1)
        
        if error_model == "unitary":
            state = init_state(n_qubit, state_name)
            state = H(n_qubit,0) @ state @ H(n_qubit,0).T.conjugate()
            state = unitary(state, n_qubit, np.sqrt(error_rate), 0)
            state = CX(n_qubit,0,1) @ state @ CX(n_qubit,0,1).T.conjugate()
            state = unitary(state, n_qubit, np.sqrt(error_rate), 0)
            state = unitary(state, n_qubit, np.sqrt(error_rate), 1)
            
        if error_model == "depolarizing&unitary":
            state = init_state(n_qubit, state_name)
            state = H(n_qubit,0) @ state @ H(n_qubit,0).T.conjugate()
            state = depolarizing(state, n_qubit, error_rate, 0)
            state = unitary(state, n_qubit, np.sqrt(error_rate), 0)
            state = CX(n_qubit,0,1) @ state @ CX(n_qubit,0,1).T.conjugate()
            state = depolarizing(state, n_qubit, error_rate, 0)
            state = depolarizing(state, n_qubit, error_rate, 1)
            state = unitary(state, n_qubit, np.sqrt(error_rate), 0)
            state = unitary(state, n_qubit, np.sqrt(error_rate), 1)
            
    return state

def X_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubit)
    P = X(n_qubit, target_idx)
    operator_0 = (I+P) / 2
    operator_1 = (I+P) / 2
    
    return operator_0, operator_1

def Y_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubit)
    P = Y(n_qubit, target_idx)
    operator_0 = (I+P) / 2
    operator_1 = (I+P) / 2
    
    return operator_0, operator_1

def Z_basis(n_qubit, target_idx):
    I = np.eye(2**n_qubit)
    P = Z(n_qubit, target_idx)
    operator_0 = (I+P) / 2
    operator_1 = (I+P) / 2
    
    return operator_0, operator_1

def pauli_measurement(n_qubit, state_name, error_model, pauli_str_list):
    target_qubit_idx_list = np.arange(n_qubit)
    pauli_meas_dict = {"X":X_basis, "Y":Y_basis, "Z":Z_basis}
    rho_0 = Bell(n_qubit, state_name, error_model, error_rate)
    rho_1 = Bell(n_qubit, state_name, error_model, error_rate)
    measurement_label_list = []
    measurement_result_list = []
    
    for target_qubit_idx, pauli_str in zip(target_qubit_idx_list, pauli_str_list):
        operator0, operator1 = pauli_meas_dict[pauli_str](n_qubit, target_qubit_idx)
        p0 = np.trace(operator0 @ rho_0)
        p1 = np.trace(operator1 @ rho_1)
        #print(f"p0 : {p0}")
        #print(f"p1 : {p1}")
        
        #rho_0 = (operator0@rho_0@operator0)/ np.sqrt(p0)
        #rho_1 = (operator1@rho_0@operator1) / np.sqrt(p1)
        
        measurement_label_list.append(pauli_str)
        measurement_result_list.append(np.random.choice(["0","1"], p=[p0,p1]))
        
    return measurement_label_list, measurement_result_list

def generate(n_qubit, state_name, each_n_shot, error_model):
    meas_pattern_list = []
    meas_label_list = []
    meas_result_list = []
    
    pauli_meas_label = ["X", "Y", "Z"]
    #operator_pattern_list = itertools.product(pauli_meas_label)
    
    for i, meas_pattern in enumerate(tqdm(itertools.product(pauli_meas_label, repeat=n_qubit))):
        meas_pattern_list.append(meas_pattern)
        print(f"measurement pattern {i} : {meas_pattern}")
        
        for j in tqdm(range(each_n_shot)):
            label, result = pauli_measurement(n_qubit, state_name, error_model, meas_pattern)
            meas_label_list.append(label)
            meas_result_list.append(result)
    
    meas_pattern_df = pd.DataFrame({"measurement_pattern":meas_pattern_list})
    meas_pattern_df["measurement_pattern"] = meas_pattern_df["measurement_pattern"].apply(lambda x: " ".join(x))
    train_df = pd.DataFrame({"measurement_label":meas_label_list, "measurement_result":meas_result_list})
    train_df["measurement_label"] = train_df["measurement_label"].apply(lambda x: " ".join(x))
    train_df["measurement_result"] = train_df["measurement_result"].apply(lambda x: " ".join(x))
    
    return meas_pattern_df, train_df

def main():
    # save train data
    meas_pattern_df, train_df = generate(n_qubit, state_name, each_n_shot, error_model)
    meas_pattern_df.to_csv(data_path+"/measurement_pattern.txt", header=False, index=False)
    train_df.to_csv(data_path+"/measurement_label.txt", columns = ["measurement_label"], header=False, index=False)
    train_df.to_csv(data_path+"/measurement_result.txt", columns = ["measurement_result"], header=False, index=False)
    
if __name__ == "__main__":
    main()