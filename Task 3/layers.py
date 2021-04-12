import numpy as np

def softmax(predictions):
    copy_predictions = np.copy(predictions)
    if predictions.ndim == 1:
        copy_predictions -= np.max(copy_predictions)
        calculated_exp = np.exp(copy_predictions)
        copy_predictions = calculated_exp / np.sum(calculated_exp)
    else:
        copy_predictions -= np.amax(copy_predictions, axis=1, keepdims=True)
        calculated_exp = np.exp(copy_predictions)
        copy_predictions = calculated_exp / np.sum(calculated_exp, axis=1, keepdims=True)
    return copy_predictions


def cross_entropy_loss(probs, target_index):
    if probs.ndim == 1:
        loss_func = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        every_batch_loss = -np.log(probs[range(batch_size), target_index])
        loss_func = np.sum(every_batch_loss) / batch_size
    return loss_func

def l2_regularization(W, reg_strength):
    l2_reg_loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W
    return l2_reg_loss, grad


def softmax_with_cross_entropy(preds, target_index):
    d_preds = softmax(preds)
    loss = cross_entropy_loss(d_preds, target_index)
    
    if preds.ndim == 1:
        d_preds[target_index] -= 1
    else:
        batch_size = preds.shape[0]
        d_preds[range(batch_size), target_index] -= 1
        d_preds /= batch_size
    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.where(X >= 0, X, 0)
        return result

    def backward(self, d_out):
        dX = np.where(self.X >= 0, 1, 0) * d_out
        return dX

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        dX = np.dot(d_out, self.W.value.T)
        dW = np.dot(self.X.T, d_out)
        dB = np.dot(np.ones((1, d_out.shape[0])), d_out)
        self.W.grad += dW
        self.B.grad += dB
        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, stride=1):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None
        self.stride = stride


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        padded_X = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, channels))

        padded_X[:, self.padding:height + self.padding, self.padding:width + self.padding, : ] = X
        self.X = padded_X

        out_height = int((width - self.filter_size + 2 * self.padding) / self.stride + 1)
        out_width = int((height - self.filter_size + 2 * self.padding) / self.stride + 1)
        
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        flat_W_shape = self.filter_size * self.filter_size * self.in_channels
        flat_W = self.W.value.reshape(flat_W_shape, self.out_channels)
        
        x_offset = 0
        for x in range(out_width):
            y_offset = 0
            for y in range(out_height):
                current_step = padded_X[:, x_offset:self.filter_size + x_offset, y_offset:self.filter_size + y_offset, :]
                flat_images = current_step.reshape(batch_size, flat_W_shape)

                current_res = np.dot(flat_images, flat_W) + self.B.value

                result_shape = result[:, x:x + 1 , y:y + 1, :].shape
                result[:, x:x + 1, y:y + 1, :] = current_res.reshape(result_shape)
                y_offset += self.stride
            x_offset += self.stride
        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        result_dX = np.zeros((batch_size, height, width, channels))
        init_shape_W = self.W.value.shape
        init_shape_B = self.B.value.shape
        flat_W_shape = self.filter_size * self.filter_size * self.in_channels

        flat_W = self.W.value.reshape(flat_W_shape, self.out_channels)

        x_offset = 0
        for x in range(out_width):
            y_offset = 0
            for y in range(out_height):
                x_from = x_offset
                x_to = self.filter_size + x_offset

                y_from = y_offset
                y_to = self.filter_size + y_offset

                x_step = self.X[:, x_from:x_to, y_from:y_to, :]
                flat_images = x_step.reshape(batch_size, flat_W_shape)
                d_step = d_out[:, x:x + 1, y:y + 1, :]
                dX = np.dot(d_step.reshape(batch_size, -1), flat_W.T)

                res_shape = x_step.shape
                result_dX[:, x_from:x_to, y_from:y_to, :] += dX.reshape(res_shape)

                dW = np.dot(flat_images.T, d_step.reshape(batch_size, -1))
                dB = np.dot(np.ones((1, d_step.shape[0])), d_step.reshape(batch_size, -1))

                self.W.grad += dW.reshape(init_shape_W)
                self.B.grad += dB.reshape(init_shape_B)

                y_offset += self.stride
            x_offset += self.stride
        
        return result_dX[:, self.padding : (height - self.padding), self.padding : (width - self.padding), :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.zeros_masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        self.zeros_masks.clear()

        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)
        
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        x_offset = 0
        for x in range(out_width):
            y_offset = 0
            for y in range(out_height):
                x_to = self.pool_size + x_offset
                y_to = self.pool_size + y_offset
                
                I = X[:, x_offset:x_to, y_offset:y_to, :]

                self.mask(x=I, pos=(x, y))
                output[:, x, y, :] = np.max(I, axis=(1, 2))
                y_offset+=self.stride
            x_offset+=self.stride
        return output
        

    def backward(self, d_out):
        _, out_height, out_width, _ = d_out.shape
        dX = np.zeros_like(self.X)

        x_offset = 0
        for x in range(out_width):
            y_offset = 0
            for y in range(out_height):
                x_to = self.pool_size + x_offset
                y_to = self.pool_size + y_offset
                
                dX[:, x_offset:x_to, y_offset:y_to, :] += d_out[:, x:x + 1, y:y + 1, :] * self.zeros_masks[(x, y)]   
                y_offset+=self.stride
            x_offset+=self.stride 
        return dX
    
    def mask(self, x, pos):
        zero_mask = np.zeros_like(x)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((batch_size, channels))
        zero_mask.reshape(batch_size, height * width, channels)[n_idx, idx, c_idx] = 1
        self.zeros_masks[pos] = zero_mask

    def params(self):
        return {}

class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        return {}
