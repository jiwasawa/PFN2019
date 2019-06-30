# momentum SGD
import numpy as np
import matplotlib.pyplot as plt
from prob2 import calc_GNN, parameter_update
TrainData_path = './machine_learning/datasets/train/'

np.random.seed(42)
D_dim = 8
T_lim = 2
# initialization of parameters
W = np.random.normal(0,0.4,(D_dim,D_dim))
A = np.random.normal(0,0.4,D_dim)
b = 0
# initialization of momentum parameters
w_W = np.zeros((D_dim,D_dim))
w_A = np.zeros(D_dim)
w_b = 0

# parameters for parameter update
epsilon = 0.001 # perturbation for gradient calculation
alpha = 0.0001 # learning rate
eta = 0.9     # momentum
num_epoch = 100
B = 10 # number of elements for mini-batch
Train_size = 1500
val_size = 500
num_mini_batch = round(Train_size/B)
train_loss_array = np.zeros(num_epoch) # training loss
train_acc_array = np.zeros(num_epoch)  # training accuracy
val_loss_array = np.zeros(num_epoch)   # validation loss
val_acc_array = np.zeros(num_epoch)    # validation accuracy


for epoch in range(num_epoch):
    bag = np.linspace(0,Train_size-1,Train_size,dtype=int)
    loss_epoch = np.zeros(num_mini_batch) # storage for mean loss for each epoch
    pred_epoch = np.zeros((Train_size,2)) # storage for [y_true, y_pred]
    
    for mini_batch in range(num_mini_batch):
        # preparation for mini-batch
        bag_index = np.linspace(0,bag.shape[0]-1,bag.shape[0],dtype=int)
        rand_index = np.random.choice(bag_index, B, replace=False)
        
        W_batch = np.zeros((D_dim,D_dim,B))
        A_batch = np.zeros((D_dim,B))
        b_batch = np.zeros(B)
        loss_batch = np.zeros(B)
        
        for k,n in enumerate(bag[rand_index]):
            G = np.loadtxt(TrainData_path+str(n)+'_graph.txt',skiprows=1)
            y_true = np.loadtxt(TrainData_path+str(n)+'_label.txt')

            loss,y_pred = calc_GNN(G,y_true,D_dim,T_lim,W,A,b)
            W_batch[:,:,k], A_batch[:,k], b_batch[k], loss_batch[k] = parameter_update(G,y_true,D_dim,T_lim,W,A,b,epsilon,alpha,loss)
            pred_epoch[mini_batch*B+k,:] = [y_true,y_pred]
            
        # update model parameters based on the mean of the mini-batch
        WW = W-alpha*np.mean(W_batch,axis=2) + eta*w_W
        AA = A-alpha*np.mean(A_batch, axis=1) + eta*w_A
        bb = b-alpha*np.mean(b_batch) + eta*w_b
        
        w_W = WW - W
        w_A = AA - A
        w_b = bb - b
        W = WW
        A = AA
        b = bb
        loss_epoch[mini_batch] = np.mean(loss_batch)
        
        # delete elements used for the mini-batch
        mask = np.ones(bag.shape[0],dtype=bool)
        mask[rand_index] = False
        bag = bag[mask]
        
    train_loss_array[epoch] = np.mean(loss_epoch)
    train_acc_array[epoch] = np.sum(pred_epoch[:,0]==pred_epoch[:,1])/Train_size
    
    
    # validation
    val_loss = np.zeros(val_size)
    val_acc = np.zeros((val_size,2))
    for i,v in enumerate(range(1500,2000)): # validation set: 1500 - 2000
        G = np.loadtxt(TrainData_path+str(v)+'_graph.txt',skiprows=1)
        y_true = np.loadtxt(TrainData_path+str(v)+'_label.txt',dtype=int)
        val_loss[i],y_pred = calc_GNN(G,y_true,D_dim,T_lim,W,A,b)
        val_acc[i,:] = [y_true,y_pred]
    val_loss_array[epoch] = np.mean(val_loss)
    val_acc_array[epoch] = np.sum(val_acc[:,0]==val_acc[:,1])/val_size
    print(str(epoch)+'th epoch finished')
    print('Validation accuracy: ',round(val_acc_array[epoch],2))

#np.save("./data/sgdm_train_loss.npy",train_loss_array)
#np.save("./data/sgdm_val_loss.npy",val_loss_array)
#np.save("./data/sgdm_train_acc.npy",train_acc_array)
#np.save("./data/sgdm_val_acc.npy",val_acc_array)
# plot results
plt.subplots(1,2,figsize=(12,4))
plt.subplot(1,2,1)
x = np.linspace(1,num_epoch,num_epoch)
plt.plot(x,train_loss_array,label='Train')
plt.plot(x,val_loss_array,label='Validation')
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.legend(loc=1,fontsize=13,frameon=False)

plt.subplot(1,2,2)
plt.plot(x,train_acc_array,label='Train')
plt.plot(x,val_acc_array,label='Validation')
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.legend(loc=4,fontsize=13,frameon=False)
plt.show()