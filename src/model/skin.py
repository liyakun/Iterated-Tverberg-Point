"""
skin data set from UCI
https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
"""

from src.lib import data, itertver, regression, test


class Skin:
    # prepare data
    my_data = data.Data()
    my_data.load_skin_data()
    X, Y = my_data.get_skin_data()
    X_train, Y_train, X_test, Y_test = my_data.split_to_train_and_test(X, Y, 7)

    def run_skin(self, number_of_training, number_of_training_instances):
        # get training weights
        weights_random = regression.Regression().gradient_descent_random_general(self.X_train, self.Y_train,
                                                                                 number_of_training,
                                                                                 number_of_training_instances)

        weights_all = regression.Regression().gradient_descent_all(self.X_train, self.Y_train)

        data_set, label = self.my_data.get_disjoint_subset_data(2000, self.X, self.Y)
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
        test.Test().perform_test(self.X_test, self.Y_test, weights_random, center_point_random,
                                 average_point_random, weights_all, "../resources/skin/result/error_random.txt")

        test.Test().perform_test(self.X_test, self.Y_test, weights_equal, center_point_equal,
                                 average_point_euqal, weights_all, "../resources/skin/result/error_equal.txt")
