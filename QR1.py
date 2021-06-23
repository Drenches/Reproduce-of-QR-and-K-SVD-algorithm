import numpy as np
from math import sqrt
import pdb
import itertools

np.set_printoptions(suppress=True)
def QR_alg(A, epsilon = 1e-10):
    A_n = []
    convergence = False
    while convergence==False:
        for i in range(1, len(A)):
            break_flag = False
            for j in range(len(A)):
                if i>j:
                    if np.abs(A[i,j])>epsilon:
                        Q, R = np.linalg.qr(A)
                        A = np.dot(R, Q)
                        A_n.append(A)
                        break_flag = True
                        break
                    else:
                        continue
            if break_flag: break
        if break_flag == True:
            continue
        else:
            convergence = True
    return A , A_n

A1 = np.array(([10,7,8,7], [7,5,6,5], [8,6,10,9],[7,5,9,10]))
dim = 5
A2 = np.zeros((dim,dim))
cap = 0
for i in range(dim):
    for j in range(dim):
        if i==j: A2[i,j] = (2 + i*2) % 10
        if i<j: A2[i,j] = (A2[i,j-1] + 1) % 10
        if i-j==1: 
            if cap==0: 
                A2[i,j]=4
            else:
                A2[i,j]= cap-1
            cap = A2[i,j]
dim = 6
A3 = np.ones((dim,dim))
for i in range(dim):
    for j in range(dim):
        A3[i,j] = 1/(j+i+1)
print(A1)
print(A2)
print(A3)
A_k1, A_n1 = QR_alg(A1)
A_k2, A_n2 = QR_alg(A2)
A_k3, A_n3 = QR_alg(A3)
eigenvalues_A_k1 = np.diag(A_k1)
eigenvalues_A_k2 = np.diag(A_k2)
eigenvalues_A_k3 = np.diag(A_k3)
pdb.set_trace()