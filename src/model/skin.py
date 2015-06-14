"""
skin data set from UCI
"""

import numpy as np

from src.lib import data, itertver, regression, test


class Skin:

    """prepare data"""
    my_data = data.Data()
    my_data.load_skin_data()
    skin_data = np.asarray(my_data.skin_data)
    np.random.shuffle(skin_data)
    X, Y = np.hsplit(skin_data, np.array([3, ]))
    Y = Y.transpose()[0]
    Y[Y == '2'] = 0
    X = X.astype(np.float)
    Y = Y.astype(np.float)

    def run_skin(self, number_of_training, number_of_training_instances):

       # get training weights
        weights_random = regression.Regression().gradient_descent_random_general(self.X, self.Y, number_of_training,
                    self.my_data.get_random_index_list(number_of_training_instances, self.X))

        weights_all = regression.Regression().gradient_descent_all(self.X, self.Y)

        data_set, label = self.my_data.get_subset_data(1000, self.X, self.Y)
        weights_equal = regression.Regression().gradient_descent_equal(data_set, label)

        # write trained weights to file
        self.my_data.write_to_csv_file("../resources/skin/result/output_weights_random.csv", weights_random)
        self.my_data.write_to_csv_file("../resources/skin/result/output_weights_equal.csv", weights_equal)
        self.my_data.write_to_csv_file("../resources/skin/result/output_weights_all.csv", weights_all)

        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)
        center_point_equal, average_point_euqal = my_center_point.get_center_and_average_point(weights_equal)

        # testing phase
        test.Test().perform_test(self.X, self.Y, weights_random, center_point_random,
                                 average_point_random, weights_all, "../resources/skin/result/error_random.txt")

        test.Test().perform_test(self.X, self.Y, weights_equal, center_point_equal,
                                 average_point_euqal, weights_all, "../resources/skin/result/error_equal.txt")

