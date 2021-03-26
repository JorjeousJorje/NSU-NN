import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """
    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output)]
        self.reg = reg
        
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
              
    def __apply_regularization(self) -> float:
        total_reg_loss = 0;
        for network_param in self.params().values():
            reg_loss, reg_grad = l2_regularization(network_param.value, self.reg)
            total_reg_loss += reg_loss
            network_param.grad += reg_grad
        return total_reg_loss

    def compute_loss_and_gradients(self, X, y):
        self.__zero_grad()
        
        last_forward_output = self.__forward_pass(X)
        loss, d_out = softmax_with_cross_entropy(last_forward_output, y)
        last_backward_dout = self.__backward_pass(d_out)
        
        loss += self.__apply_regularization()
        return loss

    def predict(self, X):
        predictions = self.__forward_pass(X)
        y_pred = np.argmax(predictions, axis = 1)
        return y_pred

    def params(self):
        result = {"FCL1_W": self.layers[0].params()['W'],
                  "FCL1_B": self.layers[0].params()['B'],
                  "FCL2_W": self.layers[2].params()['W'],
                  "FCL2_B": self.layers[2].params()['B'],}
        
        return result
