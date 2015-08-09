"""
Data Set from UCI
https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure#
"""
from src.lib.data import ProteinData, Data
from src.lib import regression, sklearnlib, itertver, test


class Protein:

    my_protein_data = ProteinData()
    my_protein_data.read_protein_file()

    def __init__(self):
        pass

    def run_protein_one_fold(self, number_of_training, number_of_training_instances, number_of_equal_disjoint_sets, fold
                             , percent_of_training):
        x_, y_ = self.my_protein_data.get_protein_data()
        x_train, x_test, y_train, y_test = sklearnlib.Sklearnlib().split_train_and_test(x_, y_, percent_of_training)

        # get training weights
        weights_random = (regression.Regression().gradient_descent_random_general(x_train, y_train, number_of_training,
                                                                                  number_of_training_instances))

        weights_all = (regression.Regression().gradient_descent_all(x_train, y_train))

        # how many "num_subset" equal
        # data_set, label = self.my_data.get_disjoint_subset_data(number_of_equal_disjoint_sets, x, y)
        data_set, label = Data().get_disjoint_subset_data(number_of_equal_disjoint_sets, x_train, y_train)

        weights_equal = (regression.Regression().gradient_descent_equal(data_set, label))

        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)
        center_point_equal, average_point_equal = my_center_point.get_center_and_average_point(weights_equal)

        # testing phase
        test.Test().perform_test(x_test, y_test, weights_random, center_point_random, average_point_random, weights_all,
                                 "../resources/protein/result_01/errors/"+str(fold) + "error_random.txt")

        test.Test().perform_test(x_test, y_test, weights_equal, center_point_equal, average_point_equal, weights_all,
                                 "../resources/protein/result_01/errors/"+str(fold) + "error_equal.txt")

    def run_protein_n_fold(self, n, number_of_training, number_of_training_instances, number_of_equal_disjoint_sets,
                           percent_of_training):
        for i in range(n):
            self.run_protein_one_fold(number_of_training, number_of_training_instances, number_of_equal_disjoint_sets, i
                                      , percent_of_training)
