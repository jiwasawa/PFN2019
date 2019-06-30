import numpy as np
import matplotlib.pyplot as plt
from prob1 import generate_graph
from prob2 import calc_GNN, parameter_update
np.random.seed(42)

D_dim = 8 # dimension for feature vector
T_lim = 2 # steps of aggregation
N = 13 # size of graph

G = generate_graph(N)
y_true = 0 # 0 or 1 # arbitrary label for graph G 

# initialization of parameters
W = np.random.normal(0,0.4,(D_dim,D_dim))
A = np.random.normal(0,0.4,D_dim)
b = 0

# parameters for parameter update
epsilon = 0.001 # perturbation for gradient calculation
alpha = 0.0001 # learning rate
num_epoch = 100

loss_array = np.zeros(num_epoch)
loss,ypred = calc_GNN(G,y_true,D_dim,T_lim,W,A,b)
for epoch in range(num_epoch):
	# update gradient parameters
    grad_W, grad_A, grad_b, new_loss = parameter_update(G,y_true,D_dim,T_lim,
                                                        W,A,b,epsilon,alpha,loss)
    loss_array[epoch] = new_loss
    loss = new_loss
    W = W - alpha*grad_W
    A = A - alpha*grad_A
    b = b - alpha*grad_b

print(G)
print('loss for epoch '+str(num_epoch)+': ',loss_array[-1]) # loss for 100th epoch
plt.plot(np.linspace(1,num_epoch,num_epoch),loss_array)
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.show()