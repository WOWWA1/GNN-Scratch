import numpy as np
import pandas as pd 
import random 

from simple_graph import Graph

class NeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size 
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(hidden_size,input_size) * np.sqrt(2/(input_size+hidden_size))
        self.W2 = np.random.randn(output_size,hidden_size) * np.sqrt(2/(hidden_size+output_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)
        
    def relu(self,x):
        return np.maximum(0,x)
    
    def cross_entropy_loss(self, predicted_probs, true_labels):

        p = np.clip(predicted_probs, 1e-15, 1 - 1e-15)

        loss = -(true_labels * np.log(p) + (1 - true_labels) * np.log(1 - p))
        return np.mean(loss) 
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    

    def forward(self, x):
        self.x = x
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.predicted_probs = self.sigmoid(self.z2)
        
        return self.predicted_probs
    
    def sigmoid_derivative(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)
    
    def loss_derivative(self,predicted_probs,true_labels):
        return predicted_probs - true_labels
    
    def backward(self,loss, true_labels):
        dldp = self.loss_derivative(self.predicted_probs,true_labels)
        dldz2 = dldp * self.sigmoid_derivative(self.z2)
        dlda1 = dldz2 * self.W2
        dldz1 = dlda1 * self.relu_derivative(self.z1)
        dldw1 = np.outer(dldz1, self.x)
        dldb1 = dldz1.flatten()
        dldw2 = np.outer(dldz2, self.a1)
        dldb2 = dldz2.flatten()
        self.update_weights(dldw1,dldb1,dldw2,dldb2)
    
    def update_weights(self,dldw1,dldb1,dldw2,dldb2):
        self.W1 -= self.learning_rate * dldw1
        self.b1 -= self.learning_rate * dldb1
        self.W2 -= self.learning_rate * dldw2
        self.b2 -= self.learning_rate * dldb2