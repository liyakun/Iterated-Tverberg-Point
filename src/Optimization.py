#! /usr/bin/pythonw
#This file contains the optimization algorithm
import numpy as np
import time

class Optimization:

    weights_all = []

    def __init__(self):
        pass

    """
    sigmoid function
    """
    def sigmoid(self, inX):
        return 1.0/(1+np.exp(-inX))

    """
    gradient ascent optional optimize algorithm
    """
    def grad_ascent(self, train_matrix, train_class_list, random_index_list_in_training, num_iteration, function):
        #  every time clean the weights_all for all training weights
        self.weights_all = []
        # get start time
        start_time = time.time()
        # get the number of rows and columns of the training matrix
        m, n = np.shape(train_matrix)
        # initialize a temporal weights vector to all 1
        weights_tmp = np.ones((n, 1))
        # iterate within the number of iteration
        for i in range(num_iteration):
            # get the index list of all the data in random_index_list_in_training
            data_index = range(len(random_index_list_in_training))
            for j in range(len(random_index_list_in_training)):
                # alpha changes on each iteration, improve oscillations
                # alpha decreases as the number of iterations increases, but it never reach 0
                alpha = 4/(1.0+j+i) + 0.001
                # randomly selecting each index from data_index, then get value from random_index_list_in_training
                # then we access the instance in train matrix to use in updating the weights, reduce periodic variations
                rand_index = int(np.random.uniform(0, len(data_index)))
                # get the sigmoid value
                h = function.sigmoid(np.sum(train_matrix[random_index_list_in_training[rand_index]]*weights_tmp))
                # compare the sigmoid value with teacher, and store the current error
                error_tmp = (train_class_list[random_index_list_in_training[rand_index]] - h)
                # store all the weights
                self.weights_all.append(np.sum(np.mat(weights_tmp)))
                # update weights in current training example
                weights_tmp += alpha * train_matrix[random_index_list_in_training[rand_index]].transpose() * error_tmp
                # remove the used instances index
                del (data_index[rand_index])
        print "\nTraining finished within %fs!\n" % (time.time() - start_time)
        return weights_tmp

    """
    Sigmoid test function
    """
    def sig_test(self, instance, weights):
        sig_value = self.sigmoid(np.sum(instance*weights))
        if sig_value > 0.5:
            return 1.0
        else:
            return 0.0
