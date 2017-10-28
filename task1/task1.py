#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np

filename = "01_homework_dataset.csv"
data = np.loadtxt(filename, delimiter=",")
print(data)

def plot2d(data):
        f = plt.figure()
        ax1 = f.add_subplot(221)
        ax1.scatter([p[0] for p in data], [p[1] for p in data], c=[p[3] for p in data])
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("XY")

        ax2 = f.add_subplot(222)
        ax2.scatter([p[0] for p in data], [p[2] for p in data], c=[p[3] for p in data])
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_title("XZ")

        ax3 = f.add_subplot(223)
        ax3.scatter([p[1] for p in data], [p[2] for p in data], c=[p[3] for p in data])
        ax3.set_xlabel("Y")
        ax3.set_ylabel("Z")
        ax3.set_title("YZ")

        f.tight_layout()

plot2d(data)
plt.show()

