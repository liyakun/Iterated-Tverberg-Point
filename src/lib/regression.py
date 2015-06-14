"""
This file contains the logistic regression algorithm
"""
from src.lib import optimization


class Regression:
    weights_random, weights_equal, weights_all = [], [], []

    def __init__(self):
        pass

    # train a logistic regression model using some optional optimize algorithm
    def gradient_descent_random(self, my_data, number_of_training, number_of_training_instances):
        for i in range(0, number_of_training):
            print i, "th training."
            reg_opt = optimization.Optimization()
            self.weights_random.append(reg_opt.gradient_descent_random(my_data.train_matrix, my_data.train_class_list,
                                                    my_data.get_random_index_list(number_of_training_instances,
                                                                                  my_data.train_matrix)))
        return self.weights_random

    def gradient_descent_equal(self, data, label):
        for i in range(len(data)):
            reg_opt = optimization.Optimization()
            self.weights_equal.append(reg_opt.gradient_descent_equal(data[i], label[i]))
        return self.weights_equal

    def gradient_descent_all(self, data_matrix, label):
        reg_opt = optimization.Optimization()
        self.weights_all.append(reg_opt.gradient_descent_equal(data_matrix, label))
        return self.weights_all

    def gradient_descent_random_general(self, data_set, label_set, number_of_training, radom_index):
        for i in range(0, number_of_training):
            print i, "th training."
            reg_opt = optimization.Optimization()
            self.weights_random.append(reg_opt.gradient_descent_random(data_set, label_set, radom_index))
        return self.weights_random
