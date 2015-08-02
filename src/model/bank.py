"""
bank data set from UCI
"""
from src.lib import test, regression
from src.lib.data import BankData, Data
from src.lib import itertver


class Bank:
    """
    filter data with unknown features && create training && testing data matrix
    """
    my_data = BankData()
    my_data.pre_process_data()

    def __init__(self):
        pass

    def run_bank(self, number_of_training, number_of_training_instances, i):
        # regression algorithm, my_data, num_of_train, num_of_training_data, num_of_iteration_each_train
        weights_random = regression.Regression().gradient_descent_random_general(self.my_data.train_matrix,
                        self.my_data.train_class_list, number_of_training, number_of_training_instances)

        weights_all = regression.Regression().gradient_descent_all(self.my_data.train_matrix,
                                                                   self.my_data.train_class_list)
        #data_set, label = Data().get_disjoint_subset_data(number_of_training, self.my_data.train_matrix,
        #                                                  self.my_data.train_class_list)
        #weights_equal = regression.Regression().gradient_descent_equal(data_set, label)
        #print len(weights_equal)

        """
        # write trained weights to file
        Data().write_to_csv_file("../resources/bank/result/output_weights_random.csv", weights_random)
        Data().write_to_csv_file("../resources/bank/result/output_weights_equal.csv", weights_equal)
        Data().write_to_csv_file("../resources/bank/result/output_weights_all.csv", weights_all)
        """
        # get center point
        my_center_point = itertver.IteratedTverberg()
        center_point_random, average_point_random = my_center_point.get_center_and_average_point(weights_random)
        #center_point_equal, average_point_euqal = my_center_point.get_center_and_average_point(weights_equal)

        # testing phase
        test.Test().perform_test(self.my_data.test_matrix, self.my_data.test_class_list, weights_random,
          center_point_random, average_point_random, weights_all, "../resources/bank/result/SecondTest/"+str(i) +
                                                                  "error_random.txt")

        #test.Test().perform_test(self.my_data.test_matrix, self.my_data.test_class_list, weights_equal,
         # center_point_equal, average_point_euqal, weights_all, "../resources/bank/result/error_equal.txt")

    def run_bank_n_fold(self, n_, number_of_training, number_of_training_instances, percent_of_train):
        for i in range(n_):
            self.my_data.split_to_train_and_test(percent_of_train)
            self.run_bank(number_of_training, number_of_training_instances, i)
