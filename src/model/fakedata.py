__author__ = 'Administrator'
from src.lib.sklearnlib import Sklearnlib
from src.lib import itertver, regression, test, data

class FakeData:

    def run_fake_data_one_fold(self, n_samples, n_features, n_informative, n_classes, percent_of_train,
                               number_of_training, number_of_training_instances, i_):

        x_, y_ = Sklearnlib().generate_fake_data(n_samples, n_features, n_informative, n_classes)
        x_train, x_test, y_train,  y_test = Sklearnlib().split_train_and_test(x_, y_, percent_of_train)
        # x_train, y_train, x_test, y_test = data.Data().split_to_train_and_test(x_, y_, 9)
        weights_random = (regression.Regression().gradient_descent_random_general(x_train.tolist(), y_train.tolist(),
                                                                                  number_of_training,
                                                                                  number_of_training_instances))

        weights_all = (regression.Regression().gradient_descent_all(x_train.tolist(), y_train.tolist()))

        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)

        # testing phase
        test.Test().perform_test(x_test.tolist(), y_test.tolist(), weights_random, center_point_random,
                                 average_point_random, weights_all,
                                 "../resources/fakedata/result/errors/"+str(i_)+"error_random.txt")

    def run_fake_data_n_fold(self,  n_samples, n_features, n_informative, n_classes, percent_of_train,
                             number_of_training, number_of_training_instances, number_of_iterations):
        for i in range(number_of_iterations):
            self.run_fake_data_one_fold(n_samples, n_features, n_informative, n_classes, percent_of_train,
                                        number_of_training, number_of_training_instances, i)
