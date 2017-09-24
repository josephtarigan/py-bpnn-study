import math
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sigmoidCharsetDictionary as cd


inputNeuronCount = 108 + 1
hiddenNeuronCount = 120 + 1
outputNeuronCount = 36

trainingCharsetsDir = '~/Workspaces/pythonenv/tensorflow/py-bpnn-study/trainingcharsets'
learningRate = 0.0450781
momentum = 0.996094
iteration = 1
epoch = 0
targetMse = 0.000001
mse = 999
mseList = np.zeros(outputNeuronCount,)

#bias1 = np.random.uniform()
#bias2 = np.random.uniform()

bias1, bias2 = 0.9, 0.9

weight1 = np.random.uniform(low=-0.5, high=0.5, size=[hiddenNeuronCount-1, inputNeuronCount]) #weights from input to layer 1, 108 + 1 bias
weight2 = np.random.uniform(low=-0.5, high=0.5, size=[outputNeuronCount, hiddenNeuronCount]) #weights from layer 1 to layer 2, 120 + 1 bias
lastDelta1 = np.zeros((hiddenNeuronCount, inputNeuronCount))
lastDelta2 = np.zeros((outputNeuronCount, hiddenNeuronCount))

def sigmoid (x) : return np.clip(1/(1 + np.exp(-x)), 0.1, 0.9)
def sigmoid_derivative (x) : return x * (1 - x)

def normalize_array(array, high, low) :
    for j in range (0, array.shape[0]) :
        for i in range (0, array.shape[1]) :
            array[j][i] = (array[j][i]-low)/high
            
while mse > targetMse :
    print ('Iteration : ' + str(iteration))
    for trainingSet in range (1, 10) :
        #print ('Charset ' + str(trainingSet))
        epoch += 1
        #print('Epoch : ' + str(epoch))
        # chars
        charIndex = 0;
        for file in os.listdir(os.path.expanduser(trainingCharsetsDir + '/' + str(trainingSet) + '/')) :
            charImg = cv.imread(os.path.expanduser(trainingCharsetsDir + '/' + str(trainingSet) + '/' + file), cv.IMREAD_GRAYSCALE)
            normalize_array(charImg, 255, 0)
        
            # feedforward
            #--------------------------- input layer -----------------------------
            inputLayerOutput = np.append([bias1,], charImg.flatten())
            #--------------------------- input layer -----------------------------
            
            #-------------------------- hidden layer -----------------------------
            hiddenLayerOutput = np.append([bias2,], sigmoid(np.dot(weight1, inputLayerOutput)).flatten())
            #-------------------------- hidden layer -----------------------------
            
            #-------------------------- output layer -----------------------------
            outputLayerOutput = sigmoid(np.dot(weight2, hiddenLayerOutput))
            #-------------------------- output layer -----------------------------
            
            # backpropagation
            #-------------------------- output layer -----------------------------
            outputError = np.subtract(cd.charsetDictionary.get(file.split('.')[0]), outputLayerOutput)
              
            #print('Local MSE, char ' + file.split('.')[0] + ' : ' + str(math.pow(np.sum(outputError), 2)/2))
            mseList[charIndex] = math.pow(np.sum(outputError), 2)/2
            
            weight2delta = (np.multiply(outputError, sigmoid_derivative(outputLayerOutput)))
            for i in range (0, outputNeuronCount) :
                #np.dot(weight2[i], outputLayerOutput[i])
                weight2[i] = np.add(weight2[i], learningRate * np.add(np.multiply(weight2delta[i], hiddenLayerOutput), (np.multiply(momentum, lastDelta2[i]))))
            #-------------------------- output layer -----------------------------
            
            #-------------------------- hidden layer -----------------------------
            hiddenNeuronOutputError = np.dot(np.transpose(weight2), weight2delta)
            weight1delta = (np.multiply(hiddenNeuronOutputError, sigmoid_derivative(hiddenLayerOutput)))
            #print(hiddenNeuronOutputError)
            #print(weight1delta)
            #input()
            for i in range (0, hiddenNeuronCount-1) :
                #np.dot(weight1[i], hiddenLayerOutput[i+1])
                weight1[i] = np.add(weight1[i], learningRate * np.add(np.multiply(weight1delta[i], inputLayerOutput), (np.multiply(momentum, lastDelta1[i+1]))))
            #-------------------------- hidden layer -----------------------------
            
            lastDelta1 = weight1delta
            lastDelta2 = weight2delta
    mse = np.sum(mseList)/outputNeuronCount
    print('MSE : ' + str(mse))  
    iteration += 1
    
# save weights
np.savetxt(os.path.expanduser("~/Workspaces/pythonenv/tensorflow/py-bpnn-study/" + "output/weight1.csv"), weight1, delimiter=",")
np.savetxt(os.path.expanduser("~/Workspaces/pythonenv/tensorflow/py-bpnn-study/" + "output/weight2.csv"), weight2, delimiter=",")

# draw MSE graph
plt.plot(mseList)
plt.ylabel('MSE')
plt.xlabel('Iteration')
plt.show()