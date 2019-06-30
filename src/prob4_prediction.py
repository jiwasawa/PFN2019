# ADAM
import numpy as np
import matplotlib.pyplot as plt
from prob2 import calc_GNN, parameter_update
TestData_path = './machine_learning/datasets/test/'

np.random.seed(42)
D_dim = 8
T_lim = 2
Test_size = 500
# initialization of parameters
W = np.load("./data/adam100_W.npy")
A = np.load("./data/adam100_A.npy")
b = np.load("./data/adam100_b.npy")

y_true = 1 # no meaning # do not need the loss anyway
y_pred_array = np.zeros(Test_size)

for ID in range(Test_size):
    G = np.loadtxt(TestData_path+str(ID)+'_graph.txt',skiprows=1)
    loss,y_pred = calc_GNN(G,y_true,D_dim,T_lim,W,A,b)
    y_pred_array[ID] = y_pred
np.savetxt('./data/prediction.txt',y_pred_array.astype(int),fmt='%i')