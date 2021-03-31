import numpy as np


class SGD:
    """
    Implements vanilla SGD update
    """
    def update(self, w, d_w, learning_rate):
        return w - d_w * learning_rate


class MomentumSGD:
    """
    Implements Momentum SGD update
    """
    def __init__(self, momentum=0.9):
        self.momentum = 0.9
        self.velocity = 0.0
    
    def update(self, w, d_w, learning_rate):
        self.velocity = self.momentum * self.velocity - learning_rate * d_w
        
        w += self.velocity
        return w
    
class Adam:
    """
    Implements Adam update
    """
    def __init__(self, beta1=0.9, beta2=0.9):
        self.accumulated = 0.0
        self.velocity = 0.0
        self.beta1 = beta1
        self.beta2 = beta2
        
    def update(self, w, d_w, learning_rate):
        self.velocity = self.beta1 * self.velocity + (1 - self.beta1) * d_w
        self.accumulated = self.beta2 * self.accumulated + (1 - self.beta2) * (d_w ** 2)
        adaptive_learning_rate = learning_rate / np.sqrt(self.accumulated)
        
        w -= self.velocity * adaptive_learning_rate
        return w
