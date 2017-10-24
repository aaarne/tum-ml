#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

filename = "01_homework_dataset.csv"
data = np.loadtxt(filename, delimiter=",")
print data
