"""
skin data set from UCI
https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
"""

from src.lib import data, itertver, regression, test
from sklearn import linear_model
from copy import deepcopy
from random import shuffle

class Skin:

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

    def run_skin_one_fold_new(self, number_of_training, number_of_training_instances, fold, percent_of_train):
        x_train, y_train, x_test, y_test, x, y = self.get_skin_data(percent_of_train)

        sgd = linear_model.SGDClassifier()
        sgd.fit(x_train, y_train)
        weights_all = deepcopy(sgd.coef_)
        weights_equal = []
        weights_random = []
        for i in xrange(number_of_training-1):
            Xs = x_train[i*number_of_training:(i+1)*number_of_training]
            ys = y_train[i*number_of_training:(i+1)*number_of_training]
            if len(ys) > 0:
                sgd = linear_model.SGDClassifier()
                sgd.fit(Xs,ys)
                w = deepcopy(sgd.coef_[0])
                weights_equal.append(w)
            Xidx = range(len(x_train))
            yidx = range(len(y_train))
            shuffle(Xidx)
            shuffle(yidx)
            Xs = x_train[Xidx[:number_of_training_instances]]
            ys = y_train[yidx[:number_of_training_instances]]
            sgd = linear_model.SGDClassifier()
            sgd.fit(Xs,ys)
            w = deepcopy(sgd.coef_[0])
            weights_random.append(w)
        weights_random = weights_random
        weights_equal = weights_equal

        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)
        center_point_equal, average_point_equal = my_center_point.get_center_and_average_point(weights_equal)

        # testing phase
        test.Test().perform_test(x_test, y_test, weights_random, center_point_random, average_point_random, weights_all,
                                 "../resources/skin/New_Setting_Result/"+str(fold) + "error_random.txt")

        test.Test().perform_test(x_test, y_test, weights_equal, center_point_equal, average_point_equal, weights_all,
                                 "../resources/skin/New_Setting_Result/"+str(fold) + "error_equal.txt")

    def run_skin_n_fold(self, number_of_training, number_of_training_instances, number_of_fold, percent_of_trains):
        for i in range(number_of_fold):
            print str(i)+"th experiment."
            self.run_skin_one_fold_new(number_of_training, number_of_training_instances, i, percent_of_trains)

