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

    #train a logistic regression model using some optional optimize algorithm
    def gradAscent(self, trainMatrix, classLabels, dataIndexMatrix, numIteration):
        regOpt = Optimization.Optimization()
        weights = regOpt.gradAscent(trainMatrix, classLabels, dataIndexMatrix, numIteration, self)
        self.weights_all = regOpt.weights_all
        return weights




