from layer import Layer
import cv2 as cv
import os
import numpy as np

trainingCharsetsDir = 'trainingcharsets'

'''rand_range = 300
x = ((np.random.random_sample() * 2)-1) * rand_range

print(np.tanh(x))'''

'''
input_weights = np.array(([1,2,3], [4,5,6]))
input_signals = np.array([1,1,1])
output = np.tanh(np.dot(input_weights, input_signals))


print(input_weights.shape)
print(input_signals.shape)
print(output.shape)

print(output)
'''

#labelSet = ()

input_signals = np.random.rand(108,) #10 input neuron, this, should come from training samples
weight1 = np.random.randn(120, 108) #weights from input to layer 1
weight2 = np.random.randn(36, 120) #weights from layer 1 to layer 2 

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

firstLayer = Layer(weight1, np.zeros(108,))

secondLayer = Layer(weight2, np.zeros(108,))


#---------------------------- training ------------------------------------

# training set
for trainingSet in range (1, 10) :
    # chars
    for file in os.listdir(trainingCharsetsDir + '/' + str(trainingSet)) :
        charImg = cv.imread(trainingCharsetsDir + '/' + str(trainingSet) + '/' + file, cv.IMREAD_GRAYSCALE)
        # feedforward
        
        #--------------------------- 1st ---------------------------------
        firstLayer.input_signals = charImg.flat
        #--------------------------- 1st ---------------------------------
        
        #--------------------------- 2nd ---------------------------------
        secondLayer.input_signals = firstLayer.calculateOutputSignals()
        #--------------------------- 2nd ---------------------------------
        
        #------------------------- output --------------------------------
        outputSignals = secondLayer.calculateOutputSignals()
        #------------------------- output --------------------------------
        
        #---------------------- error calc -------------------------------
        for outputSignal in range (0, 36) :
            error = 0
        #---------------------- error calc -------------------------------
        
#---------------------------- training ------------------------------------