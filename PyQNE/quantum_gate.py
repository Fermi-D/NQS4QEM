import numpy as np

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
    local_Y = np.array([[1,0], [0,-1]])
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
