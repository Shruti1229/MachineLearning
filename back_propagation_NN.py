import numpy as np

class two_layer_NN:
    
    def __init__(self,lr,epoch,hidden_layer_1,hidden_layer_2,w0,w_h1,w_h2):
        self.input_layer = 2
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.output_layer = 1

        self.w0 = w0        #Weights from input to 1st hidden layer
        self.w_h1 = w_h1    # Weights from 1st hidden layer to 2nd hidden layer
        self.w_h2 = w_h2    # Weights from 2nd hidden layer to output layer

        self.lr = lr
        self.epoch = epoch

    def sigmoid(self,x,w):
        return 1/(1+np.exp(-np.dot(x,w)))

    def forward_propagate(self,x):
        self.h1 = self.sigmoid(x, self.w0) # h1 = sigmoid(x * w0) --> output after first hidden layer
        self.h2 = self.sigmoid(self.h1, self.w_h1) #h2 = sigmoid(h1 * w_h1()) --> output after second hidden layer
        self.t = self.sigmoid(self.h2, self.w_h2) # output = h2 * w_h2 --> Output at last layer
        return self.t,self.h1,self.h2

    def backward_propagate(self,x,y,t):
        #dE/dW_h2(derivative of error with respect to weights of 2nd hidden layer to output later))
        delta_w_h2 = np.dot(self.h2.T,(t-y) *t*(1-t))
        backward_pass_h2 = np.dot((t-y)*t*(1-t),self.w_h2.T)*self.h2*(1-self.h2)
        #dE/dW_h1(derivative of error with respect to weights of 1st hidden layer to 2nd hidden layer))
        delta_w_h1 = np.dot(self.h1.T,backward_pass_h2)

        #dE/dW_h2(derivative of error with respect to weights of input layer to 1st hidden layer))
        backward_pass_h3 = (np.dot(backward_pass_h2,self.w_h1.T)) * self.h1 * (1-self.h1)
        delta_w0 = np.dot(x.T,backward_pass_h3)


        # Weight update of w0, w_h1 , w_h2
        self.w0 = self.w0 - self.lr * delta_w0
        self.w_h1 = self.w_h1 - self.lr* delta_w_h1
        self.w_h2 = self.w_h2 - self.lr * delta_w_h2
        return self.w0, self.w_h1, self.w_h2

    def fit(self,x,y):
        for i in range(self.epoch):

            #forward
            t,h1,h2=self.forward_propagate(x)
            #backward
            self.backward_propagate(x,y,t)    
        
        #print("loss", (np.mean(np.square(y - t))))


    def accuracy(self,x,y):
        a = []
        count = 0
        #print(x)
        t,h1,h2 = self.forward_propagate(x)
        x = [0 if i<0.5 else 1 for i in t]
        for i in range(len(x)):
            if(x[i]==y[i]):
    
                count = count + 1
        
        return count/len(x)
            


