import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN():
    def __init__(self, k, data, labels):
        self.k = k
        self.data = data
        self.labels = labels

    def predict(self, value):
        si = np.argsort([euclidean_distance(x, value) for x in self.data])
        labels = self.labels[si[0:self.k]]
        return self.labels[np.argmax(np.bincount(labels))]