import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN():
    def __init__(self, k, data, labels):
        self.k = k
        self.data = data
        self.labels = np.array(labels)

    def predict(self, value):
        si = np.argsort([euclidean_distance(x, value) for x in self.data])
        labels = self.labels[si[0:self.k]]
        return self.labels[np.argmax(np.bincount(labels))]

    def regression(self, value):
        si = np.argsort([euclidean_distance(x, value) for x in self.data])
        labels = np.array(self.labels[si[0:self.k]],dtype=float)
        return np.sum(labels)/float(self.k)

    def weighted_regression(self, value):
        distances = [euclidean_distance(x, value) for x in self.data]
        si = np.argsort(distances)
        weightsum, sum = 0.0, 0.0
        for i in range(self.k):
            weight = 1.0/distances[si[i]]
            weightsum += weight
            sum += weight * float(self.labels[si[i]])
        return sum / weightsum