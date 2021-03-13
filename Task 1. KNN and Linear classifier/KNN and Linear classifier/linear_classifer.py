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
        every_batch_loss = -np.log(probs[np.arange(batch_size), target_index.flatten()])
        loss_func = np.sum(every_batch_loss) / batch_size
    return loss_func


def softmax_with_cross_entropy(predictions, target_index):
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)
    
    # subtract the unit by the target_index to calculate gradient
    if predictions.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        dprediction[np.arange(batch_size), target_index.flatten()] -= 1
        dprediction /= batch_size
    return loss, dprediction


def l2_regularization(W, reg_strength):
    l2_reg_loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W
    
    return l2_reg_loss, grad
    

def linear_softmax(X, W, target_index):
    predictions = np.dot(X, W)
    loss, dpred = softmax_with_cross_entropy(predictions, target_index)
    # lectures formula
    dW = np.dot(X.T, dpred)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None
    
    def __generate_batches_indices(self, num_train, batch_size):
        shuffled_indices = np.arange(num_train)
        np.random.shuffle(shuffled_indices)
        sections = np.arange(batch_size, num_train, batch_size)
        batches_indices = np.array_split(shuffled_indices, sections)
        return batches_indices

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            batches_indices = self.__generate_batches_indices(num_train, batch_size)
            
            loss_sum = 0
            for batch_indices in batches_indices:
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                loss, dW = linear_softmax(batch_X, self.W, batch_y)
                reg_loss, reg_grad = l2_regularization(self.W, reg)
                
                loss += reg_loss
                dW += reg_grad
                
                self.W -= learning_rate * dW
                loss_sum += loss
                loss_history.append(loss)
            
            # num_batches = len(batches_indices)
            # avg_loss = loss_sum / num_batches
            
            # print("Epoch %i, loss: %f" % (epoch, avg_loss))

        return loss_history

    def predict(self, X):
        predictions = np.dot(X, self.W)
        y_pred = np.where(predictions == np.amax(predictions, axis=1, keepdims=True))[1]
        return y_pred



                
                                                          

            

                
