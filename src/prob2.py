import numpy as np
from prob1 import agg_loop, READOUT, generate_graph

def sigmoid(x):
    return 1/(1+np.exp(-x))

def prediction(A,b,h_G):
    """
    A, b: parameters for prediction
    s: probability of y_hat = 1
    y_hat: prediction for y
    """
    s = np.dot(A,h_G)+b
    p = sigmoid(s)
    if p>0.5:
        y_hat = 1
    else:
        y_hat = 0   
    return y_hat,s

def calc_loss(y_true,s):
    """
    calculation of loss for GNN prediction
    y_true: label for graph G
    s: probability prediction by GNN
    """
    if y_true==0:
        if s>10:
            # if exp(s)>>1, log(1+exp(s)) \sim s
            loss = s # second term of L
        else:
            loss = np.log(1+np.exp(s))
    else:
        if s<-10:
            loss = -s
        else:    
            loss = np.log(1+np.exp(-s))
    return loss

def calc_GNN(G, y_true, D, T, W, A, b):
    """
    calculate loss and prediction for graph G
    """
    x_vT = agg_loop(G, W, D, T) # result of T aggregations
    h_G = READOUT(x_vT) # h_G: feature vector of graph G

    y_pred,s = prediction(A,b,h_G)
    loss = calc_loss(y_true,s)
    return loss,y_pred

def parameter_update(G,y_true,D,T,W,A,b,epsilon,alpha,loss_orig):
    """
    update parameter W, A, b based on the calculated gradient
    """
    # gradient for W
    grad_W = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            delta_W = np.zeros((D,D))
            delta_W[i,j] = epsilon*1
            loss2,ypred2 = calc_GNN(G,y_true,D,T,W+delta_W,A,b)
            
            grad_Wi = (loss2-loss_orig)/epsilon
            grad_W[i,j] = grad_Wi
    
    # gradient for A
    grad_A = np.zeros(D)
    for i in range(D):
        delta_A = np.zeros(D)
        delta_A[i] = epsilon*1
        loss2,ypred2 = calc_GNN(G,y_true,D,T,W,A+delta_A,b)
        
        grad_Ai = (loss2-loss_orig)/epsilon
        grad_A[i] = grad_Ai
    
    # gradient for b
    delta_b = epsilon*1
    loss2,ypred2 = calc_GNN(G,y_true,D,T,W,A,b+delta_b)
    grad_b = (loss2-loss_orig)/epsilon
    
    W2 = W - alpha*grad_W
    A2 = A - alpha*grad_A
    b2 = b - alpha*grad_b
    new_loss,new_ypred = calc_GNN(G,y_true,D,T,W2,A2,b2)
    return grad_W, grad_A, grad_b, new_loss