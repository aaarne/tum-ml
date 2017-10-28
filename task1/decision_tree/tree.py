import numpy as np

class Split(object):
	def __init__(self, dim, value):
		self._dim = dim
		self._value = value

	def isLeft(self, value):
		return value[self._dim] <= self._value

	def isRight(self, value):
		return not self.isLeft(value)

class DecisionTreeNode(object):
	def __init__(self, classlabel):
		self.classlabel = classlabel
		self.isLeaf = True

	def addSplit(self, split, leftclass, rightclass):
		self.isLeaf = False
		self.split = split
		self.left = DecisionTreeNode(leftclass)
		self.right = DecisionTreeNode(rightclass)

class DecisionTree(object):
	def __init__(self):
		self.root = None

	def computeTree(self, maxDepth=2):
		pass