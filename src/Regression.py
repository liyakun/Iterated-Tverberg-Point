#! /usr/bin/pythonw
#This file contains the logistic regression algorithm

import numpy as np
import Optimization

class Regression:
    """
    logistic regression algorithm
    """

    weights_all = []

    def __init__(self):
        pass

    #calculate the sigmoid function
    def sigmoid(self, inX):
        return 1.0/(1+np.exp(-inX))

    #train a logistic regression model using some optional optimize algorithm
    def gradAscent(self, trainMatrix, classLabels, dataIndexMatrix, numIteration):
        regOpt = Optimization.Optimization()
        weights = regOpt.gradAscent(trainMatrix, classLabels, dataIndexMatrix, numIteration, self)
        #regOpt.plotWeights()
        self.weights_all = regOpt.weights_all
        return weights




