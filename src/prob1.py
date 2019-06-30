import numpy as np

# graph G will be read as an adjacency matrix
# functions: agg1, agg2, agg_loop, READOUT, generate_graph

def agg1(G, x_v):
    """
    aggregation-1 x_v \cdot G
    G: N*N numpy array (graph)
    D: Dimension of feature vector
    x_v: feature vector
    """
    a_v = np.dot(x_v,G)
    
    return a_v

def agg2(a_v, W):
    """
    aggregation-2 ReLU(W \cdot a_v)
    a_v = result of aggregation-1
    D: dimension of feature vector
    W: D*D matrix for aggregation
    """
    aa_v = np.dot(W,a_v)
    x_vt = np.maximum(0,aa_v) # applying ReLU
    
    return x_vt

def agg_loop(G, W, D, T):
    """
    loop of aggregation-1,2 for T times
    G: N*N numpy array (graph)
    W: D*D matrix for aggregation
    D: Dimension of feature vector
    The first element of the feature vector is 1, others: 0
    """
    # constructing initial feature vector
    N = G.shape[0] # number of vertices of graph G
    x_v0 = np.zeros((D,N)) 
    x_v0[0,:] = 1
    
    x_v = x_v0 # initialization of feature vector
    for t in range(T):
        a_v = agg1(G, x_v=x_v)
        x_vt = agg2(a_v, W=W)
        x_v = x_vt
    return x_v

def READOUT(x_vT):
    """
    summation of feature vectors after T aggregations
    """
    h_G = np.sum(x_vT,axis=1)
    return h_G

def generate_graph(N):
    """
    generate a N*N random symmetric matrix 
    all elements are 0 or 1
    diagonal elements are 0
    """
    m = 0.5*np.random.rand(N,N)
    mm = m.T + m
    mm = np.round(mm)
    for i in range(N):
        mm[i,i] = 0
    return mm
