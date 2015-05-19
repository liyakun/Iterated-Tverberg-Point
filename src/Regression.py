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

    """
    train a logistic regression model using some optional optimize algorithm
    """
    def grad_ascent(self, num_train, train_matrix, train_class_list, random_index_list_in_training, num_iteration):
        for i in range(0, num_train):
            print i, "th training."
            reg_opt = Optimization.Optimization()
            self.weights.append(reg_opt.grad_ascent(train_matrix, train_class_list,  random_index_list_in_training,
                                                    num_iteration, reg_opt))
            self.weights_all.append(reg_opt.weights_all)
        return self.weights, self.weights_all




