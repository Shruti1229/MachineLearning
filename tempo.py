from back_propagation_NN import two_layer_NN
import numpy as np
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import ParameterGrid
from sklearn.model_selection import ParameterGrid
x = np.array(([2, 2], [3, 3], [4, 4],[5, 5],[1,1], [2, 3], [2, 4],[2, 5], [2, 6], [2, 7]), dtype=float)
y = np.array(([0], [0], [0],[0], [0], [1],[1], [1], [1],[1]), dtype=float)
x = x/np.amax(x, axis=0)


#set hyperparameters 

param_grid = {'lr':[0.1,0.01], 'epoch':[10,100], 'hidden_layer_1': [20, 30], 'hidden_layer_2' : [10,8]}

grid = ParameterGrid(param_grid)
best_score = -1

#testing dataset
x_test = np.array(([6,6],[2,2],[3,2],[4,4],[2,3]),dtype=float)
y_test = np.array(([0],[0],[1],[0],[1]),dtype = float)

#shape of input and output layer
input_layer = x.shape[1]
output_layer = y.shape[1]

#grid search on learning rate, number of epochs, number of node in first hidden layer, number of nodes in second layer
for params in grid:

    w0 = np.random.randn(input_layer,params['hidden_layer_1'])                 #Weights from input to 1st hidden layer
    w_h1 = np.random.randn(params['hidden_layer_1'],params['hidden_layer_2'])  # Weights from 1st hidden layer to 2nd hidden layer
    w_h2 = np.random.randn(params['hidden_layer_2'],output_layer)              # Weights from 2nd hidden layer to output layer
    
    tn = two_layer_NN(params['lr'], params['epoch'], params['hidden_layer_1'], params['hidden_layer_2'],w0,w_h1,w_h2)
    
    tn.fit(x,y)
    score = tn.accuracy(x_test,y_test)
    
    if score > best_score:
        best_score = score
        learning_rate = params['lr']
        number_epochs = params['epoch']
        nodes_hidden_1 =params['hidden_layer_1']
        nodes_hidden_2 =params['hidden_layer_2']


print("The best accuracy of {0:2.2f}% is obtained at \n Epoch : {1} \n Learning Rate : {2} \n Number of nodes in first hidden layer : {3}".format(best_score*100, number_epochs, learning_rate, nodes_hidden_1))
print(" Number of nodes in second hidden layer {0}".format(nodes_hidden_2))
