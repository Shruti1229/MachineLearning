from back_propagation_NN import two_layer_NN
import numpy as np


x = np.array(([2, 2], [3, 3], [4, 4],[5, 5],[1,1], [2, 3], [2, 4],[2, 5], [2, 6], [2, 7]), dtype=float)
y = np.array(([0], [0], [0],[0], [0], [1],[1], [1], [1],[1]), dtype=float)
x = x/np.amax(x)


lr = 0.01
epoch = 100
hidden_layer_1 = 2
hidden_layer_2 = 3
input_layer = x.shape[1]
output_layer = y.shape[1]


w0 = np.random.randn(input_layer,hidden_layer_1)            #Weights from input to 1st hidden layer
w_h1 = np.random.randn(hidden_layer_1,hidden_layer_2)       # Weights from 1st hidden layer to 2nd hidden layer
w_h2 = np.random.randn(hidden_layer_2,output_layer)         #Weights from 1st hidden layer to 2nd hidden layer

tn = two_layer_NN(lr, epoch, hidden_layer_1, hidden_layer_2,w0,w_h1,w_h2)


#initialize all weights with zeros
w0_zeros = np.zeros((input_layer,hidden_layer_1)) 
w_h1_zeros = np.zeros((hidden_layer_1,hidden_layer_2)) 
w_h2_zeros = np.zeros((hidden_layer_2,output_layer))
tn_zeros = two_layer_NN(lr, epoch, hidden_layer_1, hidden_layer_2,w0_zeros,w_h1_zeros,w_h2_zeros)

#initialize all weights with ones
w0_ones = np.ones((input_layer,hidden_layer_1)) 
w_h1_ones = np.ones((hidden_layer_1,hidden_layer_2)) 
w_h2_ones = np.ones((hidden_layer_2,output_layer))
tn_ones = two_layer_NN(lr, epoch, hidden_layer_1, hidden_layer_2,w0_ones,w_h1_ones,w_h2_ones)



def sigmoid_test():
    
    x = 0.5
    w = 0.1
    a = tn.sigmoid(x,w)
    print(a)
    
    x = 0.8
    w = 0
    a = tn.sigmoid(x,w)
    print(a)

    x = 0.8
    w = 1
    a = tn.sigmoid(x,w)
    print(a)    

def test_forward_propagate():
    x_test = np.array(([6,6],[2,2],[3,2]),dtype=float)
    output,h1,h2 = tn.forward_propagate(x_test)
    print("Predicted value without training",output)
    flag = 0
    tn.fit(x,y)
    output_fit,h1,h2 = tn.forward_propagate(x_test)
    print("Predicted value after training",output_fit)
   

    for i in range(len(output)):
        if(output_fit[i] != output[i]):
            flag = 1
    # Raise Error if the Model was not trained (i.e no weihjts were updated)
    assert flag !=0 , "Model was not trained"

def test_back_propagate():
    t,h1,h2 = tn_zeros.forward_propagate(x)
    w0,w_h1,w_h2=tn_zeros.backward_propagate(x,y,t)
    #when all weights are initilized as zeros, after one epoch all updated weights should be zeros
    print(w0)
    print(w_h1)
    print(w_h2)
    
    #when all weights are initilized as ones, after one epoch all updated weights should be nearly equal to one
    t,h1,h2 = tn_ones.forward_propagate(x)
    w0,w_h1,w_h2=tn_ones.backward_propagate(x,y,t)
    print(w0)
    print(w_h1)
    print(w_h2)
    
    


    

sigmoid_test()
test_forward_propagate()
test_back_propagate()
