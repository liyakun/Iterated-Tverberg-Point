#! /usr/bin/pythonw
#This file contains the main function
import Test, Data, Plot, Regression

weights, weights_all = [], []

#filter data with unknown features && create dummy mat matrix
myData = Data.Data()
myData.dataReady()

#training with regression algorithm
weights, weights_all = Regression.Regression().gradAscent(3, myData.train_matrix, myData.train_class_list,
                                                          myData.getRandomIndexList(1500), 500)

#write trained weights to file
myData.writeToCsvFile("../resources/output_weights.csv", weights)

#plot weights convergence from regression
Plot.Plot().plot(weights_all)

#testing phase
Test.Test().performTest(1, myData.test_matrix, myData.test_class_list, weights)
