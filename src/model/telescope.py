"""
model process telescope dataset
https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
"""
from src.lib.data import TelescopeData, Data
from src.lib import sklearnlib, regression, test, itertver
import os


class Telescope:

    my_telescope = TelescopeData()
    my_telescope.read_telescope_data()

    def __init__(self):
        pass

    def run_telescope_one_fold(self, number_of_training, number_of_training_instances, number_of_equal_disjoint_sets,
                               fold, percent_of_training, path):
        x_, y_ = self.my_telescope.get_telescope_data()
        x_train, x_test, y_train, y_test = sklearnlib.Sklearnlib().split_train_and_test(x_, y_, percent_of_training)

        # get training weights
        weights_random = (regression.Regression().gradient_descent_random_general(x_train, y_train, number_of_training,
                                                                                  number_of_training_instances))

        weights_all = (regression.Regression().gradient_descent_all(x_train, y_train))

        # how many "num_subset" equal
        # data_set, label = self.my_data.get_disjoint_subset_data(number_of_equal_disjoint_sets, x, y)
        # data_set, label = Data().get_disjoint_subset_data(number_of_equal_disjoint_sets, x_train, y_train)

        # weights_equal = (regression.Regression().gradient_descent_equal(data_set, label))

        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)
        # center_point_equal, average_point_equal = my_center_point.get_center_and_average_point(weights_equal)

        # testing phase
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path))

        test.Test().perform_test(x_test, y_test, weights_random, center_point_random, average_point_random, weights_all,
                                 path+str(fold) + "error_random.txt")

        # test.Test().perform_test(x_test, y_test, weights_equal, center_point_equal, average_point_equal, weights_all,
        #                         "../resources/protein/result/"+str(fold) + "error_equal.txt")

    def run_telescope_n_fold(self, n, number_of_training, number_of_training_instances, number_of_equal_disjoint_sets,
                             percent_of_training, path):
        for i in range(n):
            self.run_telescope_one_fold(number_of_training, number_of_training_instances, number_of_equal_disjoint_sets, i
                                      , percent_of_training, path)


