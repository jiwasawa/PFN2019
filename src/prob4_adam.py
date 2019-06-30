# ADAM
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
# initialization of ADAM parameters
m_W = np.zeros((D_dim,D_dim))
m_A = np.zeros(D_dim)
m_b = 0

s_W = np.zeros((D_dim,D_dim))
s_A = np.zeros(D_dim)
s_b = 0 

# parameters for parameter update
epsilon = 0.001 # perturbation for gradient calculation
alpha = 0.001 # learning rate

beta1 = 0.9
beta2 = 0.999
epsilon2 = 10**(-8) # constant to prevent divergence

num_epoch = 100
B = 10 # number of elements for mini-batch
Train_size = 1500
val_size = 500
num_mini_batch = round(Train_size/B)
train_loss_array = np.zeros(num_epoch) # training loss
train_acc_array = np.zeros(num_epoch)  # training accuracy
val_loss_array = np.zeros(num_epoch)   # validation loss
val_acc_array = np.zeros(num_epoch)    # validation accuracy

for t,epoch in enumerate(range(num_epoch)):
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
            
        # update model parameters based on the mean of the mini-batch using ADAM
        g_W = np.mean(W_batch,axis=2)
        g_A = np.mean(A_batch, axis=1)
        g_b = np.mean(b_batch)
        
        m_W = beta1*m_W + (1-beta1)*g_W
        s_W = beta2*s_W + (1-beta2)*np.power(g_W,2)
        m_A = beta1*m_A + (1-beta1)*g_A
        s_A = beta2*s_A + (1-beta2)*np.power(g_A,2)
        m_b = beta1*m_b + (1-beta1)*g_b
        s_b = beta2*s_b + (1-beta2)*np.power(g_b,2)
        
        m_hat_W = m_W/(1-beta1**t)
        s_hat_W = s_W/(1-beta2**t)
        m_hat_A = m_A/(1-beta1**t)
        s_hat_A = s_A/(1-beta2**t)
        m_hat_b = m_b/(1-beta1**t)
        s_hat_b = s_b/(1-beta2**t)
        
        W = W - alpha*m_hat_W/(np.sqrt(s_hat_W)+epsilon2)
        A = A - alpha*m_hat_A/(np.sqrt(s_hat_A)+epsilon2)
        b = b - alpha*m_hat_b/(np.sqrt(s_hat_b)+epsilon2)

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
#np.save("./data/adam100_W.npy",W)
#np.save("./data/adam100_A.npy",A)
#np.save("./data/adam100_b.npy",b)
#np.save("./data/adam_train_loss.npy",train_loss_array)
#np.save("./data/adam_val_loss.npy",val_loss_array)
#np.save("./data/adam_train_acc.npy",train_acc_array)
#np.save("./data/adam_val_acc.npy",val_acc_array)
# plot results
plt.subplots(1,2,figsize=(12,4))
plt.subplot(1,2,1)
x = np.linspace(1,num_epoch,num_epoch)
plt.plot(x,train_loss_array,label="Train")
plt.plot(x,val_loss_array,label="Validation")
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.legend(loc="best",frameon=False,fontsize=13)

plt.subplot(1,2,2)
plt.plot(x,train_acc_array,label="Train")
plt.plot(x,val_acc_array,label="Validation")
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.legend(loc="best",frameon=False,fontsize=13)
#plt.savefig('./data/adam_training.png',bbox_inches='tight',dpi=100)
plt.show()