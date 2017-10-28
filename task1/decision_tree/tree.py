import numpy as np
import matplotlib.pyplot as plt


class Split(object):
    def __init__(self, dim, value):
        self._dim = dim
        self._value = value

    def is_left(self, value):
        return value[self._dim] <= self._value

    def is_right(self, value):
        return not self.is_left(value)


class DecisionTreeNode(object):
    def __init__(self, classlabel):
        self.classlabel = classlabel
        self.isLeaf = True

    def _add_split(self, split, data, classes):
        self.isLeaf = False
        self.split = split
        leftclasses = np.array([classes[i] for i in range(len(classes)) if split.is_left(data[i])])
        rightclasses = np.array([classes[i] for i in range(len(classes)) if split.is_right(data[i])])
        self.left = DecisionTreeNode(np.bincount(leftclasses).argmax())
        self.right = DecisionTreeNode(np.bincount(rightclasses).argmax() if len(rightclasses)>0 else 0)
        return leftclasses, rightclasses

    def gini_index(self, data, classes):
        gini = 1.0
        for classlabel in np.unique(classes):
            d = (np.count_nonzero(classes == classlabel) / len(classes)) ** 2
            gini -= d
        return gini

    def generate_delta_gini_distribution(self, data, classes, dim):
        for value in data[:,dim]:
            n = len(classes)
            split = Split(dim, value)
            leftdata = np.array([row for row in data if split.is_left(row)])
            rightdata = np.array([row for row in data if split.is_right(row)])
            tmpnode = DecisionTreeNode(self.classlabel)
            leftclasses, rightclasses = tmpnode._add_split(split, data, classes)
            yield tmpnode.gini_index(data, classes) \
                - (len(leftdata) / n) * tmpnode.left.gini_index(leftdata, leftclasses) \
                - (len(rightdata) / n) * tmpnode.right.gini_index(rightdata, rightclasses)

    def add_leaves(self, data, classes):
        if len(np.unique(classes)) == 1:
            raise ValueError()
        ndim = np.size(data, axis=1)
        maxindices, maxvalues = list(), list()
        for dim in range(ndim):
            dist = np.array([v for v in self.generate_delta_gini_distribution(data, classes, dim)])
            tit = "Distribution along dim {}".format(dim)
            if True:
                plt.figure()
                plt.title(tit)
                s = np.argsort(data[:,dim])
                plt.plot(data[:,dim][s], dist[s])
            maxindices.append(dist.argmax())
            maxvalues.append(dist.max())
        dim = np.argmax(maxvalues)
        index = maxindices[dim]
        print("Decided to split in dimension {} at value {}.".format(dim, data[index, dim]))
        self._add_split(Split(dim, data[index,dim]), data, classes)
        leftindices = [index for index in range(len(data)) if self.split.is_left(data[index])]
        rightindices = [index for index in range(len(data)) if self.split.is_right(data[index])]
        print(classes[leftindices], classes[rightindices])
        return leftindices, rightindices


class DecisionTree(object):
    def __init__(self, data, classes, maxDepth=2):
        self.root = self.compute_tree(data, classes, maxDepth)

    def compute_tree(self, data, classes, maxDepth):
        root = DecisionTreeNode(np.bincount(classes).argmax())
        self.recur(root, data, classes, maxDepth)
        return root

    def recur(self, node, data, classes, maxDepth, n=0):
        if n>=maxDepth:
            return
        else:
            l, r = node.add_leaves(data, classes)
            try:
                self.recur(node.left, data[l], classes[l], maxDepth, n+1)
            except ValueError:
                pass

            try:
                self.recur(node.right, data[r], classes[r], maxDepth, n+1)
            except ValueError:
                pass

    def predict(self, value):
        return self.predict_recur(self.root, value)

    def predict_recur(self, node, value):
        if node.isLeaf:
            return node.classlabel
        else:
            return self.predict_recur(node.left if node.split.is_left(value) else node.right, value)