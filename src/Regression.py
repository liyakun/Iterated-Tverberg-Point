#! /usr/bin/pythonw
#This file contains the logistic regression algorithm
import Optimization

class Regression:
    """
    logistic regression algorithm
    """
    weights = []
    weights_all = []

    def __init__(self):
        pass

    #train a logistic regression model using some optional optimize algorithm
    def gradAscent(self, num_train, trainMatrix, classLabels, dataIndexMatrix, numIteration):
        for i in range(0, num_train):
            print i, "th training."
            regOpt = Optimization.Optimization()
            self.weights.append(regOpt.gradAscent(trainMatrix, classLabels, dataIndexMatrix, numIteration,
                                                  regOpt))
            self.weights_all.append(regOpt.weights_all)
        return self.weights, self.weights_all




