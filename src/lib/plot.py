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

    def file_to_list(self, file_open):
        line_list = [line for line in file_open.readlines()]
        temp_list, center_list, average_list, all_list = [], [], [], []
        length = len(line_list)
        for i,  line in enumerate(line_list):
            if i <= (length-4):
                temp_list.append(float(line.split(":")[1].rstrip()))
            elif i == length-3:
                average_list.append(float(line.split(":")[1].rstrip()))
            elif i == length-2:
                center_list.append(float(line.split(":")[1].rstrip()))
            else:
                all_list.append(float(line.split(":")[1].rstrip()))
        return temp_list, average_list, center_list, all_list

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
        ax.scatter(mean_point[0], mean_point[1], mean_point[2], color='blue', marker='o')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        plt.show()

    def plot_error(self, path):
        fr = open(path)
        temp_list = []
        for line in fr:
                temp_list.append(line.split(":")[1])

        with pltpage.PdfPages(path+".weights.pdf") as pdf:
            fig = plt.figure()
            plt.title("error")
            plt.plot(temp_list, marker='.', linestyle='--', color='blue', markeredgecolor='red')
            pdf.savefig(fig)
            plt.close()

    def box_plot_equal(self, equal_list, equal_special_list, path):
        fig1 = plt.figure(1, figsize=(10, 6))
        ax = fig1.add_subplot(111)
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
        bp_0 = ax.boxplot(equal_list, 1, meanprops=meanlineprops, meanline=True, showmeans=True)

        bp_1 = ax.boxplot(equal_special_list[0])
        for median in bp_1['medians']:
            median.set(color='red', linewidth=2)

        bp_2 = ax.boxplot(equal_special_list[1])
        for median in bp_2['medians']:
            median.set(color='blue', linewidth=2)

        bp_3 = ax.boxplot(equal_special_list[2])
        for median in bp_3['medians']:
            median.set(color='yellow', linewidth=2)

        # Remove top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        ax.set_xlabel('Equal Sample-Lists of Weight Vector')
        ax.set_ylabel('Errors of Each Weight Vector')

        # change outline color, fill color and linewidth of the boxes
        for box in bp_0['boxes']:
            # change outline color
            box.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the whiskers
        for whisker in bp_0['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the caps
        for cap in bp_0['caps']:
            cap.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the medians
        for median in bp_0['medians']:
            median.set(color='#b2df8a', linewidth=2)

        fig1.savefig(path+'fig_equal.png', bbox_inches='tight')

        plt.close()

    def box_plot_random(self, random_list, random_special_list, path):

        fig2 = plt.figure(1, figsize=(10, 6))
        ax = fig2.add_subplot(111)
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
        meanlineprops_0 = dict(linestyle='--', linewidth=2.5, color='purple')
        bp_0 = ax.boxplot(random_list, 1, meanprops=meanlineprops_0, meanline=True, showmeans=True)

        bp_1 = ax.boxplot(random_special_list[0])
        for median in bp_1['medians']:
            median.set(color='red', linewidth=2)

        bp_2 = ax.boxplot(random_special_list[1])
        for median in bp_2['medians']:
            median.set(color='blue', linewidth=2)

        bp_3 = ax.boxplot(random_special_list[2])
        for median in bp_3['medians']:
            median.set(color='yellow', linewidth=2)

        # Remove top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        # Hide these grid behind plot objects
        ax.set_axisbelow(True)
        ax.set_xlabel('Random Sample-Lists of Weight Vector')
        ax.set_ylabel('Errors of Each Weight Vector')

        # change outline color, fill color and linewidth of the boxes
        for box in bp_0['boxes']:
            # change outline color
            box.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the whiskers
        for whisker in bp_0['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the caps
        for cap in bp_0['caps']:
            cap.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the medians
        for median in bp_0['medians']:
            median.set(color='#b2df8a', linewidth=2)

        for median in bp_0['medians']:
            median.set(color='#b2df8a', linewidth=2)

        fig2.savefig(path+'fig_random.png', bbox_inches='tight')
        plt.close()

    def box_plot(self, n, path):
        equal_list, random_list,  = [], []
        equal_special_list, random_special_list = [[], [], []], [[], [], []]
        for i in range(n):
            fr_equal = (open(path+str(i)+"error_equal.txt"))
            fr_random = (open(path+str(i)+"error_random.txt"))
            equal, eq_average_list, eq_center_list, eq_all_list = self.file_to_list(fr_equal)
            random, ra_average_list, ra_center_list, ra_all_list = self.file_to_list(fr_random)
            equal_list.append(equal)
            equal_special_list[0].append(eq_average_list)
            equal_special_list[1].append(eq_center_list)
            equal_special_list[2].append(eq_all_list)
            random_list.append(random)
            random_special_list[0].append(ra_average_list)
            random_special_list[1].append(ra_center_list)
            random_special_list[2].append(ra_all_list)

        print len(random_special_list)
        self.box_plot_equal(equal_list, equal_special_list, path)
        self.box_plot_random(random_list, random_special_list, path)
