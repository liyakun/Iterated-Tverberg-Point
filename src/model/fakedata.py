__author__ = 'Administrator'
import os
from src.lib.sklearnlib import Sklearnlib
from src.lib import itertver, regression, test, data

class FakeData:

    x_, y_ = '', ''
    def run_fake_data_one_fold(self, n_samples, n_features, n_informative, n_classes, percent_of_train,
                               number_of_training, number_of_training_instances, i_, path, str_):
        x_train, x_test, y_train,  y_test = Sklearnlib().split_train_and_test(self.x_, self.y_, percent_of_train)
        # x_train, y_train, x_test, y_test = data.Data().split_to_train_and_test(x_, y_, 9)
        weights_random = (regression.Regression().gradient_descent_random_general(x_train.tolist(), y_train.tolist(),
                                                                                  number_of_training,
                                                                                  number_of_training_instances))

        weights_all = (regression.Regression().gradient_descent_all(x_train.tolist(), y_train.tolist()))

        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)

        # testing phase
        if str_ == "features":
            if not os.path.exists(path+"errors/"+str(n_features)+"/"):
                os.makedirs(os.path.dirname(path+"errors/"+str(n_features)+"/"))

            test.Test().perform_test(x_test.tolist(), y_test.tolist(), weights_random, center_point_random,
                                     average_point_random, weights_all, path+"errors/"+str(n_features)+"/"+str(i_) +
                                     "error_random.txt")
        else:
            if not os.path.exists(path+"errors/"+str(number_of_training)+"/"):
                os.makedirs(os.path.dirname(path+"errors/"+str(number_of_training)+"/"))

            test.Test().perform_test(x_test.tolist(), y_test.tolist(), weights_random, center_point_random,
                                     average_point_random, weights_all, path+"errors/"+str(number_of_training)+"/" +
                                     str(i_)+"error_random.txt")

    def run_fake_data_n_fold(self, n_samples, n_features, n_informative, n_classes, percent_of_train,
                             number_of_training, number_of_training_instances, number_of_iterations, path, str_):
        for i in range(number_of_iterations):
            self.run_fake_data_one_fold(n_samples, n_features, n_informative, n_classes, percent_of_train,
                                        number_of_training, number_of_training_instances, i, path, str_)

    def dimension_test(self, n_samples, fetures, n_informative, n_classes, percent_of_train,
                             number_of_training, number_of_training_instances, number_of_iterations):
        for i in range(2, fetures):
            self.run_fake_data_n_fold(n_samples, i, i, n_classes, percent_of_train,
                             number_of_training, number_of_training_instances, number_of_iterations,
                                      "../resources/fakedata/result/dimensions/", "features")

    def vectors_test(self, n_samples, features, n_informative, n_classes, percent_of_train,
                             number_of_training, number_of_training_instances, number_of_iterations):
        self.x_, self.y_ = Sklearnlib().generate_fake_data(n_samples, features, n_informative, n_classes)
        while number_of_training >= 1000:
            self.run_fake_data_n_fold(n_samples, features, n_informative, n_classes, percent_of_train,
                             number_of_training, number_of_training_instances, number_of_iterations,
                                      "../resources/fakedata/result/instances/", ",")
            number_of_training -= 500


    # def instances_test(self, n_samples, features, n_informative, n_classes, percent_of_train,
    #                          number_of_training, number_of_training_instances, number_of_iterations):
    #     while n_samples >= 1000:
    #         self.run_fake_data_n_fold(n_samples, features, n_informative, n_classes, percent_of_train,
    #                          number_of_training, number_of_training_instances, number_of_iterations,
    #                                   "../resources/fakedata/result/instances/")
    #         n_samples -= 500
