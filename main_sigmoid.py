import os
import sigmoidCharsetDictionary as cd
import cv2 as cv
import numpy as np
import math

inputNeuronCount = 108 + 1
hiddenNeuronCount = 120 + 1
outputNeuronCount = 36

trainingCharsetsDir = 'trainingcharsets'
learningRate = 0.150781
momentum = 0.986328
iteration = 1
epoch = 0
targetMse = 0.000001
mse = 999
mseList = np.zeros(outputNeuronCount,)

bias1 = np.random.uniform()
bias2 = np.random.uniform()

weight1 = np.random.uniform(low=-0.5, high=0.5, size=[hiddenNeuronCount-1, inputNeuronCount]) #weights from input to layer 1, 108 + 1 bias
weight2 = np.random.uniform(low=-0.5, high=0.5, size=[outputNeuronCount, hiddenNeuronCount]) #weights from layer 1 to layer 2, 120 + 1 bias
lastDelta1 = np.zeros((hiddenNeuronCount, inputNeuronCount))
lastDelta2 = np.zeros((outputNeuronCount, hiddenNeuronCount))

def sigmoid (x) : return 1/(1 + np.exp(-x))
def sigmoid_derivative (x) : return x * (1 - x)

def normalize_array(array, high, low) :
    for j in range (0, array.shape[0]) :
        for i in range (0, array.shape[1]) :
            array[j][i] = (array[j][i]-low)/high
            
while mse > targetMse :
    print ('Iteration : ' + str(iteration))
    for trainingSet in range (1, 10) :
        print ('Charset ' + str(trainingSet))
        # chars
        charIndex = 0;
        for file in os.listdir(trainingCharsetsDir + '/' + str(trainingSet)) :
            charImg = cv.imread(trainingCharsetsDir + '/' + str(trainingSet) + '/' + file, cv.IMREAD_GRAYSCALE)
            normalize_array(charImg, 255, 0)
        
            # feedforward
            #--------------------------- input layer -----------------------------
            inputLayerOutput = np.append([bias1,], charImg.flatten())
            #--------------------------- input layer -----------------------------
            
            #-------------------------- hidden layer -----------------------------
            hiddenLayerOutput = np.append([bias2,], sigmoid(np.dot(weight1, inputLayerOutput)).flatten())
            if (file.split('.')[0] is '0') : 
                print(hiddenLayerOutput)
            #-------------------------- hidden layer -----------------------------
            
            #-------------------------- output layer -----------------------------
            outputLayerOutput = sigmoid(np.dot(weight2, hiddenLayerOutput))
            #-------------------------- output layer -----------------------------
            
            # backpropagation
            #-------------------------- output layer -----------------------------
            outputError = np.subtract(outputLayerOutput, cd.charsetDictionary.get(file.split('.')[0]))
            print('Local MSE, char ' + file.split('.')[0] + ' : ' + str(math.pow(np.sum(outputError), 2)/2))
            mseList[charIndex] = math.pow(np.sum(outputError), 2)/2
            
            weight2delta = (learningRate * (outputError * sigmoid_derivative(np.clip(outputLayerOutput, 0.00001, 0.99999))))
            '''print(outputLayerOutput)
            print(sigmoid_derivative(np.clip(outputLayerOutput, 0.00001, 0.99999)))
            print(outputError)
            print(weight2delta)'''
            #input()
            for i in range (0, outputNeuronCount) :
                '''print(weight2delta[i])
                print(weight2[i])'''
                weight2[i] += weight2delta[i]
                '''print(weight2[i])
                input()'''
            #-------------------------- output layer -----------------------------
            
            #-------------------------- hidden layer -----------------------------
            hiddenNeuronOutputError = np.dot(np.transpose(weight2), weight2delta)
            weight1delta = (learningRate * (hiddenNeuronOutputError * sigmoid_derivative(np.clip(hiddenLayerOutput, 0.00001, 0.99999))))
            for i in range (0, hiddenNeuronCount-1) :
                weight1[i] += weight1delta[i+1]
            #-------------------------- hidden layer -----------------------------
            
            lastDelta1 = weight1delta
            lastDelta2 = weight2delta
    print('MSE : ' + str(np.sum(mseList)/outputNeuronCount))  
    iteration += 1    