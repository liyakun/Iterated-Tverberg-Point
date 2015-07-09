"""
skin data set from UCI
https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
"""

from src.lib import data, itertver, regression, test

class Skin:
    # prepare data
    my_data = data.Data()
    my_data.load_skin_data()

    def __init__(self):
        pass

    # each run will get different subset from X, Y, as the matrix are shuffled before they are split
    def get_skin_data(self, percent_of_train):
        x, y = self.my_data.get_skin_data()
        x_train, y_train, x_test, y_test = self.my_data.split_to_train_and_test(x, y, percent_of_train)
        return x_train, y_train, x_test, y_test, x, y

    def run_skin_one_fold(self, number_of_training, number_of_training_instances, fold, percent_of_train,
                          number_of_equal_disjoint_sets):

        # get skin data
        x_train, y_train, x_test, y_test, x, y = self.get_skin_data(percent_of_train)

        # get training weights
        weights_random = (regression.Regression().gradient_descent_random_general(x_train, y_train, number_of_training,
                                                                                  number_of_training_instances))

        weights_all = (regression.Regression().gradient_descent_all(x_train, y_train))

        # how many "num_subset" equal
        # data_set, label = self.my_data.get_disjoint_subset_data(number_of_equal_disjoint_sets, x, y)
        data_set, label = self.my_data.get_disjoint_subset_data(number_of_equal_disjoint_sets, x_train, y_train)

        weights_equal = (regression.Regression().gradient_descent_equal(data_set, label))

        # write trained weights to file
        self.my_data.write_to_csv_file("../resources/skin/result/weights/output_weights_random"+str(fold)+".csv",
                                       weights_random)
        self.my_data.write_to_csv_file("../resources/skin/result/weights/output_weights_equal"+str(fold)+".csv",
                                       weights_equal)
        self.my_data.write_to_csv_file("../resources/skin/result/weights/output_weights_all"+str(fold)+".csv",
                                       weights_all)
        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)
        center_point_equal, average_point_equal = my_center_point.get_center_and_average_point(weights_equal)

        # testing phase
        test.Test().perform_test(x_test, y_test, weights_random, center_point_random, average_point_random, weights_all,
                                 "../resources/skin/result/errors/"+str(fold) + "error_random.txt")

        test.Test().perform_test(x_test, y_test, weights_equal, center_point_equal, average_point_equal, weights_all,
                                 "../resources/skin/result/errors/"+str(fold) + "error_equal.txt")

    def run_skin_n_fold(self, number_of_training, number_of_training_instances, number_of_fold, percent_of_train,
                        number_of_equal_disjoint_sets):
        for i in range(number_of_fold):
            print str(i)+"th experiment."
            self.run_skin_one_fold(number_of_training, number_of_training_instances, i, percent_of_train,
                                   number_of_equal_disjoint_sets)

