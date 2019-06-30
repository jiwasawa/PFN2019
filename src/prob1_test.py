from prob1 import generate_graph,agg1,agg2,agg_loop, READOUT
import numpy as np

graph_size = 3
D = 2 # dimension of feature vector
T = 1 # steps of aggregation 

G = generate_graph(graph_size)
print('Graph G: ')
print(G)

x_v = np.zeros((D,G.shape[0])) 
x_v[0,:] = 1

W = np.ones((D,D))
a_v = agg1(G,x_v)
print('\nResult for aggregation-1 for graph G: ')
print(a_v)

x_vt = agg2(a_v,W)
print('\nResult for aggregation-2 for graph G: ')
print(x_vt)


x_vT = agg_loop(G,W,D,T)
print('\nResult for T='+str(T)+' aggregation loops: ')
print('(if T=1, the result should coincide with the result for aggregation-2 above)')
print(x_vT)

h_G = READOUT(x_vT)
print('\nResult for READOUT h_G: ')
print(h_G)

#####################################################################
print("\nTest for ReLU")

G = generate_graph(3)
print('Graph G: ')
print(G)

x_v = np.zeros((D,G.shape[0])) 
x_v[0,:] = 1

W = -np.ones((D,D)) # to check whether ReLU is working or not
a_v = agg1(G, x_v)
print('\nResult for aggregation-1 for graph G: ')
print(a_v)

x_vt = agg2(a_v, W)
print('\nResult for aggregation-2 for graph G: ')
print(x_vt)


x_vT = agg_loop(G,W,D,T)
print('\nResult for T='+str(T)+' aggregation loops: ')
print('(if T=1, the result should coincide with the result for aggregation-2 above)')
print(x_vT)

h_G = READOUT(x_vT)
print('\nResult for READOUT h_G: ')
print(h_G)