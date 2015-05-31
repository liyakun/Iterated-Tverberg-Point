"""
This file provide plot tools
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltpage
from mpl_toolkits.mplot3d import  Axes3D

class Plot:

    def __init__(self):
        pass

    def plot(self, weights_all):
        size = len(weights_all)
        with pltpage.PdfPages("../resources/pic/weights.pdf") as pdf:
            for i in xrange(0, size):
                fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                plt.title("%dth weights convergence" % i)
                plt.plot(weights_all[i])
                pdf.savefig(fig)
                plt.close()

    def plot3dpoints(self, points, coefficients, mean_point):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        _x, _y, _z = [], [], []
        for i in range(len(points)):
            _x.append(points[i][0])
            _y.append(points[i][1])
            _z.append(points[i][2])

        ax.scatter(_x, _y, _z, c='r', marker='o', color='yellow')
        ax.scatter(coefficients[0], coefficients[1], coefficients[2], color='red', marker='o')
        # ax.scatter(mean_point[0], mean_point[1], mean_point[2], color='red', marker='o')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        plt.show()
