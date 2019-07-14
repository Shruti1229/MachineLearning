import numpy as np
from back_propagation_NN import two_layer_NN


tn = two_layer_NN()
x = np.array(([2, 2], [3, 3], [4, 4],[5, 5],[1,1], [2, 3], [2, 4],[2, 5], [2, 6], [2, 7]), dtype=float)
y = np.array(([0], [0], [0],[0], [0], [1],[1], [1], [1],[1]), dtype=float)
x = x/np.amax(x, axis=0)
tn.train(x,y)
