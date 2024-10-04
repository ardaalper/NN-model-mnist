import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist


(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtrain = Xtrain.reshape(Xtrain.shape[0], -1) / 255.0  
Xtest = Xtest.reshape(Xtest.shape[0], -1) / 255.0  

def Encoder(y, n):
    return np.eye(n)[y]

YtrainEncode = Encoder(Ytrain, 10)
YtestEncode = Encoder(Ytest, 10)

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return (x > 0) * 1

def softmax(x):
    a = np.exp(x - np.max(x, axis=1, keepdims=True))
    return a / np.sum(a, axis=1, keepdims=True)

imageSize = 784   
hidden_1_size = 64
hidden_2_size = 64 
outputSize = 10   

W1 = np.random.randn(imageSize, hidden_1_size) * np.sqrt(1. / imageSize)
b1 = np.zeros((1, hidden_1_size))

W2 = np.random.randn(hidden_1_size, hidden_2_size) * np.sqrt(1. / hidden_1_size)
b2 = np.zeros((1, hidden_2_size))

W3 = np.random.randn(hidden_2_size, outputSize) * np.sqrt(1. / hidden_2_size)
b3 = np.zeros((1, outputSize))

learningSpeed = 0.01 
epochs = 30 
batch = 128


for epoch in range(epochs):

    for i in range(0, Xtrain.shape[0], batch):
        Xbatch = Xtrain[i:i + batch]
        ybatch = YtrainEncode[i:i + batch]


        z1 = np.dot(Xbatch, W1) + b1
        a1 = relu(z1) 

        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)  

        z3 = np.dot(a2, W3) + b3
        a3 = softmax(z3)

        dz3 = a3 - ybatch
        dW3 = np.dot(a2.T, dz3) / batch
        db3 = np.sum(dz3, axis=0, keepdims=True) / batch

        dz2 = np.dot(dz3, W3.T) * reluDerivative(a2) 
        dW2 = np.dot(a1.T, dz2) / batch
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch

        dz1 = np.dot(dz2, W2.T) * reluDerivative(a1)  
        dW1 = np.dot(Xbatch.T, dz1) / batch
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch

        W3 -= learningSpeed * dW3
        b3 -= learningSpeed * db3

        W2 -= learningSpeed * dW2
        b2 -= learningSpeed * db2

        W1 -= learningSpeed * dW1
        b1 -= learningSpeed * db1

    z1 = np.dot(Xtrain, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)

    predictions = np.argmax(a3, axis=1)
    dogruluk = np.mean(predictions == Ytrain)

    print(f"tur: {epoch + 1}/{epochs}, Dogruluk: {dogruluk:.4f}")

z1 = np.dot(Xtest, W1) + b1
a1 = relu(z1)

z2 = np.dot(a1, W2) + b2
a2 = relu(z2)

z3 = np.dot(a2, W3) + b3
a3 = softmax(z3)

predictions = np.argmax(a3, axis=1)
dogruluk = np.mean(predictions == Ytest)
print(f"Test Accuracy: {dogruluk:.4f}")

np.savetxt('W1.txt', W1, delimiter=',')
np.savetxt('b1.txt', b1, delimiter=',')
np.savetxt('W2.txt', W2, delimiter=',')
np.savetxt('b2.txt', b2, delimiter=',')
np.savetxt('W3.txt', W3, delimiter=',')
np.savetxt('b3.txt', b3, delimiter=',')