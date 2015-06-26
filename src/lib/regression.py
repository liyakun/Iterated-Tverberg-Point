"""
This file contains the logistic regression algorithm
"""
from src.lib import optimization
from src.lib import data

class Regression:

    def __init__(self):
        pass

    # train a logistic regression model using some optional optimize algorithm
    def gradient_descent_random(self, my_data, number_of_training, number_of_training_instances):
        weights_random = []
        for i in range(0, number_of_training):
            print i, "th training."
            reg_opt = optimization.Optimization()
            weights_random.append(reg_opt.gradient_descent_random(my_data.train_matrix, my_data.train_class_list,
                                                                  my_data.get_random_index_list(
                                                                      number_of_training_instances, my_data.train_matrix
                                                                  )))
        return weights_random

    def gradient_descent_equal(self, data, label):
        weights_equal = []
        for i in range(len(data)):
            reg_opt = optimization.Optimization()
            weights_equal.append(reg_opt.gradient_descent_equal(data[i], label[i]))
        return weights_equal

    def gradient_descent_all(self, data_matrix, label):
        weights_all = []
        reg_opt = optimization.Optimization()
        weights_all.append(reg_opt.gradient_descent_equal(data_matrix, label))
        return weights_all

    def gradient_descent_random_general(self, data_set, label_set, number_of_training, number_of_training_instances):
        weights_random = []
        my_data = data.Data()
        for i in range(0, number_of_training):
            print i, "th training."
            reg_opt = optimization.Optimization()
            weights_random.append(reg_opt.gradient_descent_random(data_set, label_set,
                                                                  my_data.get_random_index_list(
                                                                      number_of_training_instances, data_set)))
        return weights_random
