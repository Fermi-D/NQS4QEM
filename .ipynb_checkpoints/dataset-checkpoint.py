import os
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

import gate
import measurement

def generate(state, n_qubit, error_model, each_n_shot):
    meas_pattern_list = []
    meas_label_list = []
    meas_result_list = []
    
    pauli_meas_label = ["X", "Y", "Z"]
    
    for i, meas_pattern in enumerate(tqdm(itertools.product(pauli_meas_label, repeat=n_qubit))):
        meas_pattern_list.append(meas_pattern)
        print(f"measurement pattern {i+1}/{3**n_qubit} : {meas_pattern}")
        
        for j in tqdm(range(each_n_shot)):
            label, result = measurement.pauli(state, n_qubit, error_model, meas_pattern)
            meas_label_list.append(label)
            meas_result_list.append(result)
    
    meas_pattern_df = pd.DataFrame({"measurement_pattern":meas_pattern_list})
    meas_pattern_df["measurement_pattern"] = meas_pattern_df["measurement_pattern"].apply(lambda x: " ".join(x))
    train_df = pd.DataFrame({"measurement_label":meas_label_list, "measurement_result":meas_result_list})
    train_df["measurement_label"] = train_df["measurement_label"].apply(lambda x: " ".join(x))
    train_df["measurement_result"] = train_df["measurement_result"].apply(lambda x: " ".join(x))
    
    return meas_pattern_df, train_df

def save(meas_pattern_df, train_df, train_data_path):
    os.makedirs(train_data_path, exist_ok = True)
    meas_pattern_df.to_csv(train_data_path+"measurement_pattern.txt", header=False, index=False)
    train_df.to_csv(train_data_path+"measurement_label.txt", columns = ["measurement_label"], header=False, index=False)
    train_df.to_csv(train_data_path+"measurement_result.txt", columns = ["measurement_result"], header=False, index=False)