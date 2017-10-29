#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from decision_tree import DecisionTree
from mpl_toolkits.mplot3d import Axes3D
from knn import KNN

def load_data():
    filename = "01_homework_dataset.csv"
    raw_data = np.loadtxt(filename, delimiter=',')
    data = raw_data[:,:-1]
    classes = np.array(raw_data[:,-1], dtype=int)
    print("Data:{}".format(data))
    print("Classes:{}".format(classes))
    return data, classes


def plot2d(data, classes):
        f = plt.figure()
        ax1 = f.add_subplot(221)
        ax1.scatter([p[0] for p in data], [p[1] for p in data], c=[classlabel for classlabel in classes])
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("XY")
        ax1.set_aspect('equal')

        ax2 = f.add_subplot(222)
        ax2.scatter([p[0] for p in data], [p[2] for p in data], c=[classlabel for classlabel in classes])
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_title("XZ")
        ax2.set_aspect('equal')

        ax3 = f.add_subplot(223)
        ax3.scatter([p[1] for p in data], [p[2] for p in data], c=[classlabel for classlabel in classes])
        ax3.set_xlabel("Y")
        ax3.set_ylabel("Z")
        ax3.set_title("YZ")
        ax3.set_aspect('equal')

        f.tight_layout()
        return f


def plot3d(data, classes):
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.scatter([p[0] for p in data], [p[1] for p in data], [p[2] for p in data], c=[classlabel for classlabel in classes])
    return f

if __name__ == '__main__':
    data, classes = load_data()
    f = plot3d(data, classes)
    f = plot2d(data, classes)
    tree = DecisionTree(data, classes, maxDepth=2)

    v1 = np.array([4.1, -0.1, 2.2])
    v2 = np.array([6.1, 0.4, 1.3])

    print("Prediction for {}: {}".format(v1, tree.predict(v1)))
    print("Prediction for {}: {}".format(v2, tree.predict(v2)))

    newdata = list()
    newclasses = list()
    for x1 in np.linspace(0, 10, 10):
        for x2 in np.linspace(-1, 1, 10):
            for x3 in np.linspace(0, 6, 10):
                v = np.array([x1, x2, x3])
                newdata.append(v)
                newclasses.append(tree.predict(v))
    f2 = plot3d(newdata, newclasses)

    tree.print_tree()

    print()
    print()
    print()

    print("KNN Classifier")
    knn = KNN(3, data, classes)
    for x in [v1, v2]:
        print("KNN prediction for {}: {}".format(x, knn.predict(x)))

    print("KNN Weighted Regression")
    for x in [v1, v2]:
        print("Weighted KNN regression for {}: {}".format(x, knn.weighted_regression(x)))

    plt.show()


