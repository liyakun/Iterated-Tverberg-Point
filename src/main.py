#! /usr/bin/pythonw
#This file contains the main function
import csv
import Regression
import Data
import Plot

weights = []
weights_all = []

#filter data with unknown features && create dummy mat matrix
myData = Data.Data()
myData.loadDataSet()
myData.filterDataSet()
myData.convertAttrToMatrix()
myData.splitToTrainAndTest()

#regression algorithm
myRegression = Regression.Regression()
for i in range(0, 15):
    print i, "th training."
    weights.append(myRegression.gradAscent(myData.train_matrix, myData.train_class_list, myData.getRandomIndexList(1500), 500))
    weights_all.append(myRegression.weights_all)

#write trained weights to file
with open("../resources/output_weights.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(weights)

#plot weights convergence from regression
myPlot = Plot.Plot()
myPlot.plot(weights_all)

#testing phase
