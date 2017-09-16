import numpy as np

class Layer () :
    
    def __init__(self, input_weights, input_signals) :
        self.input_weights = input_weights
        self.input_signals = input_signals
    
    def calculateOutputSignals (self) :
        return np.tanh(np.dot(self.input_weights, self.input_signals))