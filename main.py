import os
import charsetDictionary as cd
import cv2 as cv
from layer import Layer
import numpy as np
import math

#print(input_signals)
#print(weight1)

#--------------------------- 1st ---------------------------------
#firstLayer = Layer(weight1, input_signals)
#layer1OutputSignal = firstLayer.calculateOutputSignals();
#--------------------------- 1st ---------------------------------

#--------------------------- 2nd ---------------------------------
#secondLayer = Layer(weight2, layer1OutputSignal)
#layer2OutputSignal = secondLayer.calculateOutputSignals()
#--------------------------- 2nd ---------------------------------

#print(layer1OutputSignal.shape)
#print(layer1OutputSignal)

#print(layer2OutputSignal.shape)
#print(layer2OutputSignal)

#img1 = cv.imread('trainingcharsets/1/a.jpg', cv.IMREAD_GRAYSCALE)
#print(type(img1))
#print(img1.shape)

'''
cv.imshow('image', img1)
cv.waitKey()
cv.destroyAllWindows()
'''

#print(os.path.splitext('trainingcharsets/1/a.jpg')[0])
#splitsize = len((trainingCharsetsDir + '/1/a.jpg').split('/'))
#print((trainingCharsetsDir + '/1/a.jpg').split('/')[splitsize-1].split('.')[0])

#------------------------------ init --------------------------------------

inputNeuronCount = 108 + 1
hiddenNeuronCount = 120 + 1
outputNeuronCount = 36

weight1 = np.random.randn(hiddenNeuronCount-1, inputNeuronCount) #weights from input to layer 1, 108 + 1 bias
weight2 = np.random.randn(outputNeuronCount, hiddenNeuronCount) #weights from layer 1 to layer 2, 120 + 1 bias
delta1 = np.zeros((hiddenNeuronCount, inputNeuronCount))
lastDelta1 = np.zeros((hiddenNeuronCount, inputNeuronCount))
delta2 = np.zeros((outputNeuronCount, hiddenNeuronCount))
lastDelta2 = np.zeros((outputNeuronCount, hiddenNeuronCount))
layer1Error = np.zeros(hiddenNeuronCount)
layer2Error = np.zeros(outputNeuronCount)

layer1 = Layer(weight1, np.zeros(inputNeuronCount,))
layer2 = Layer(weight2, np.zeros(hiddenNeuronCount,))

inputLayerOutput = np.zeros(inputNeuronCount)
layer1Output = np.zeros(hiddenNeuronCount)
layer2Output = np.zeros(outputNeuronCount)

trainingCharsetsDir = 'trainingcharsets'
learningRate = 0.050781
momentum = 0.986328
iteration = 1

#input_signals = np.random.rand(108,) #10 input neuron, this, should come from training samples
bias1 = np.random.random()
bias2 = np.random.random()
epoch = 0
targetMse = 0.000001
mse = 999;
mseList = np.zeros(outputNeuronCount,)

def normalize_array(array, high, low) :
    for j in range (0, array.shape[0]) :
        for i in range (0, array.shape[1]) :
            array[j][i] = (array[j][i]-low)/high

#---------------------------- training ------------------------------------

# training set
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
            #--------------------------- 1st ---------------------------------
            inputLayerOutput = np.zeros(inputNeuronCount)
            inputLayerOutput[0] = bias1
            index = 1
            for item in charImg.flat :
                inputLayerOutput[index] = item
                index += 1
    
            layer1.input_signals = inputLayerOutput
            #--------------------------- 1st ---------------------------------
            
            #--------------------------- 2nd ---------------------------------
            layer1Output[0] = bias1
            index = 1
            for item in layer1.calculateOutputSignals() :
                layer1Output[index] = item
                index += 1
            
            layer2.input_signals = layer1Output
            #--------------------------- 2nd ---------------------------------
            
            #------------------------- output --------------------------------
            layer2Output = layer2.calculateOutputSignals()
            #------------------------- output --------------------------------
            
            #---------------------- error calc -------------------------------
            for index in range (0, outputNeuronCount) :
                #print(cd.charsetDictionary.get(file.split('.')[0]))
                #print('Desired : ' + str(str(cd.charsetDictionary.get(file.split('.')[0])[index])) + ', Output : ' + str(layer2Output[index]))
                layer2Error[index] = (cd.charsetDictionary.get(file.split('.')[0])[index]) - layer2Output[index]
                #print('Error : ' + str(layer2Error[index]))

            print('Local MSE : ' + str(math.pow(np.sum(layer2Error), 2)/2))
            mseList[charIndex] = math.pow(np.sum(layer2Error), 2)/2
            charIndex += 1;
            #---------------------- error calc -------------------------------

            #--------------------- weights 2 delta calc ----------------------
            for j in range (0, outputNeuronCount) :
                for i in range (0, hiddenNeuronCount) :
                    '''if weight2[j][i] == 1 :
                        weight2[j][i] = 0.9999
                    elif weight2[j][i] == -1 :
                        weight2[j][i] = -0.9999
                    
                    if layer2Output[j] == 1 :
                        layer2Output[j] = 0.9999
                    elif layer2Output[j] == -1 :
                        layer2Output[j] = -0.9999
                        
                    if layer2Error[j] == 1 :
                        layer2Error[j] = 0.9999
                    elif layer2Error[j] == -1 :
                        layer2Error[j] = -0.9999'''
                        
                    # calculate sum
                    sum_output = np.dot(weight2[j,], layer1Output) + bias2
                        
                    delta2[j][i] = (learningRate * (layer2Error[j] * (1 - math.pow(math.tanh(sum_output), 2)) * layer2Output[j])) + (momentum * lastDelta2[j][i])
                    #print('Delta Layer 2 [' + str(j) + '][' + str(i) + '] : ' + str(delta2[j][i]))
                    #print(str(layer2Error[j]) + ' * ' + str((1 - math.pow(math.tanh(sum_output), 2))) + ' * ' + str(layer2Output[j])) 
            #--------------------- weights 2 delta calc ----------------------

            #--------------------- weights 2 weight adj ----------------------
            for j in range (0, outputNeuronCount) :
                for i in range (0, hiddenNeuronCount) :
                    weight2[j][i] = weight2[j][i] + delta2[j][i]
            #--------------------- weights 2 weight adj ----------------------  
  
            #--------------------- layer 1 error calc ------------------------
            layer1Error = np.dot(np.transpose(weight2), layer2Error)
            #--------------------- layer 1 error calc ------------------------

            #--------------------- weights 1 delta calc ----------------------
            for j in range (0, hiddenNeuronCount-1) :
                for i in range (0, inputNeuronCount) :
                    '''if weight1[j][i] == 1 :
                        weight1[j][i] = 0.9999
                    elif weight1[j][i] == -1 :
                        weight1[j][i] = -0.9999
                    
                    if layer1Output[j] == 1 :
                        layer1Output[j] = 0.9999
                    elif layer1Output[j] == -1 :
                        layer1Output[j] = -0.9999
                        
                    if layer1Error[j] == 1 :
                        layer1Error[j] = 0.9999
                    elif layer1Error[j] == -1 :
                        layer1Error[j] = -0.9999'''
                    
                    # calculate sum
                    sum_output = np.dot(weight1[j,], inputLayerOutput)
                    
                    delta1[j][i] = (learningRate * (layer1Error[j+1] * (1 - math.pow(math.tanh(sum_output), 2)) * layer1Output[j+1])) + (momentum * lastDelta1[j][i])
                    #print('Delta Layer 1 [' + str(j) + '][' + str(i) + '] : ' + str(delta1[j][i]))
            #--------------------- weights 1 delta calc ----------------------

            #--------------------- weights 1 weight adj ----------------------
            for j in range (0, hiddenNeuronCount-1) :
                for i in range (0, inputNeuronCount) :
                    weight1[j][i] = weight1[j][i] + delta1[j][i]
            #--------------------- weights 1 weight adj ----------------------
            
            #--------------------- weight preserving -------------------------
            lastDelta1 = delta1
            lastDelta2 = delta2
            #--------------------- weight preserving -------------------------

    print('MSE : ' + str(np.sum(mseList)/outputNeuronCount))
    iteration += 1
#---------------------------- training ------------------------------------

# save weights
np.savetxt("weight1.csv", weight1, delimiter=",")
np.savetxt("weight2.csv", weight2, delimiter=",")