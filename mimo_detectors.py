import numpy as np

def MF(H):
# Matched Filter signal detector
# H is channel matrix N x K, y is received vector N x 1
# Returns A_H matrix N x N
    H_H = np.matrix(H).H
    K = H.shape[1]
    norm_factors = np.ones(K)
    for i in range(K):
        norm_factors[i] = 1/(np.linalg.norm(H_H[i][:]))**2
    A_H = np.matmul(np.diag(norm_factors),H_H)
    return A_H

def ZF(H):
# Zero Forcing signal detector
# H is channel matrix N x K, y is received vector N x 1
# # Returns A_H matrix N x N
    H_H = np.matrix(H).H
    M = np.matmul(H_H,H)
    M_inv = np.linalg.inv(M)
    A_H = np.matmul(M_inv,H_H)
    return A_H

def LMMSE(H,N0,es):
# Linear Minimum Mean Square Error signal detector
# H is channel matrix N x K, y is received vector N x 1, N0/2 is variance of noise at receiver
# Returns A_H matrix N x N
    H_H = np.matrix(H).H
    K = H.shape[1]
    N = H.shape[0]
    M = np.matmul(H_H,H) + (N0/es)*np.eye(K)
    M_inv = np.linalg.inv(M)
    A_H = np.matmul(M_inv,H_H)
    return A_H