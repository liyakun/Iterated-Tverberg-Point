#! /usr/bin/pythonw
#This file contains the logistic regression algorithm

from numpy import *
import Optimization

class Regression:
    """
    logistic regression algorithm
    """

    def __init__(self):
        pass

    #calculate the sigmoid function
    def sigmoid(self, inX):
        return 1.0/(1+exp(-inX))

    #train a logistic regression model using some optional optimize algorithm
    def gradAscent(self, dataMatrix, classLabels, numIteration):
        regOpt = Optimization.Optimization()
        return regOpt.gradAscent(dataMatrix, classLabels, numIteration, self)





