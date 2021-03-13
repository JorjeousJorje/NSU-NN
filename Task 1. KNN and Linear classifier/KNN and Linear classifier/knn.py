import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        # Нам пришло 16 фотографий из тестового (итогового) набора данных
        # Теперь совственно и нужно посчитать расстояния от каждого семп-
        # ла в test до уже выученных train!
        
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
        return dists

    def compute_distances_one_loop(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            
            # "np.abs(X[i_test] - self.train_X)" сделает numpy broadcasting
            # axis = 1 позволяет сложить 121 раз 121-ну строку размером 3072
            # что и будет являться расстоянием
            dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis = 1)
        return dists
    
    def compute_distances_no_loops(self, X):
        # X[:, np.newaxis] makes X as 3 dim array of 16 (1, 3027) raws
        # self.train_X[np.newaxis, :] makes self.train_X as 3 dim array
        # of (127, 3072) matrix (matrix goes 3D)
        
        # Substraction of those arrays happens as broadcasting every raw
        # of X to matrix self.train_X. And it happens 16 times. 
        # After np.abs substraction we have shape=(16, 121, 3072)
        # "axis = 2" allows us to sum throught columns of (121, 3072) matrix
        
        dists = np.sum(np.abs(X[:, np.newaxis] - self.train_X[np.newaxis, :]), axis = 2)
        return dists
        

    def predict_labels_binary(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # argpartition возворащает массив индексов
            # возвращает так, если бы я его сначала отсортирвоал, 
            # вернёт индексы k наименьших элементов(расстояний) в исходном массиве
            k_min_distances = dists[i].argpartition(self.k - 1)
            neighbours = self.train_y[k_min_distances[: self.k]]
            values, counts = np.unique(neighbours, return_counts=True)
            pred[i] = values[np.argmax(counts)]
        return pred

    def predict_labels_multiclass(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            k_min_distances = dists[i].argpartition(self.k - 1)
            neighbours = self.train_y[k_min_distances[: self.k]]
            values, counts = np.unique(neighbours, return_counts=True)
            pred[i] = values[np.argmax(counts)]
        return pred

