import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy)

class ConvNet:

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):

        self.layers = [ConvolutionalLayer(in_channels=input_shape[2], out_channels=input_shape[2], filter_size=conv1_channels, padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, stride=2),
                       ConvolutionalLayer(in_channels=input_shape[2], out_channels=input_shape[2], filter_size=conv2_channels, padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, stride=2),
                       Flattener(),
                       FullyConnectedLayer(n_input=192, n_output=n_output_classes)]


    def __zero_grad(self):
        for network_param in self.params().values():
            network_param.grad.fill(0.0)
        
    def __forward_pass(self, X: np.array) -> np.array:
        last_forward_output = X
        for layer in self.layers:
            last_forward_output = layer.forward(last_forward_output)
        return last_forward_output
    
    def __backward_pass(self, d_out: np.array) -> np.array:
        last_backward_dout = d_out
        for layer in reversed(self.layers):
            last_backward_dout = layer.backward(last_backward_dout)
        return last_backward_dout
    
    def compute_loss_and_gradients(self, X, y):
        self.__zero_grad()
        last_forward_output = self.__forward_pass(X)
        loss, d_out = softmax_with_cross_entropy(last_forward_output, y)
        last_backward_dout = self.__backward_pass(d_out)
        return loss

    def predict(self, X):
        predictions = self.__forward_pass(X)
        y_pred = np.argmax(predictions, axis = 1)
        return y_pred

    def params(self):
        result = {  'W1': self.layers[0].params()['W'],         'B1': self.layers[0].params()['B'], 
                    'W2': self.layers[3].params()['W'],         'B2': self.layers[3].params()['B'], 
                    'W3': self.layers[7].params()['W'],         'B3': self.layers[7].params()['B']}
        return result
