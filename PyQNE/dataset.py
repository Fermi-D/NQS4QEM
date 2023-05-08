import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import itertools
import quantum_gate as gate
import projection_operator as meas_operator
import target_circuit as circuit
import measurement

def generate(n_qubit, n_shot):
    meas_pattern_list = []
    meas_label_list = []
    meas_result_list = []
    
    pauli_meas_label = ["X", "Y", "Z"]
    meas_pattern_list = itertools.product(pauli_meas_label)
    
    for meas_pattern in tqdm(meas_pattern_list):
        meas_pattern_list.append(meas_pattern)
        print(f"measurement pattern : {meas_pattern}")
        
        for i in tqdm(range(n_shot)):
            label, result = measurement(n_qubit, meas_pattern)
            meas_label_list.append(label)
            meas_result_list.append(result)
    
    meas_pattern_df = pd.DataFrame({"measurement_pattern":meas_pattern_list})
    meas_pattern_df["measurement_pattern"] = meas_pattern_df["measurement_pattern"].apply(lambda x: " ".join(x))
    train_df = pd.DataFrame({"measurement_label":meas_label_list, "measurement_result":meas_result_list})
    train_df["measurement_label"] = train_df["measurement_label"].apply(lambda x: " ".join(x))
    train_df["measurement_result"] = train_df["measurement_result"].apply(lambda x: " ".join(x))
    
    return meas_pattern_df, train_df

"""
def data_export(meas_pattern_df, train_df):
    meas_pattern_df.to_csv("./{}/data/measurement_pattern.txt", header=False, index=False)
    train_df.to_csv("./data//measurement_label.txt", columns = ["measurement_label"], header=False, index=False)
    train_df.to_csv("./data/{}/measurement_result.txt", columns = ["measurement_result"], header=False, index=False)
"""