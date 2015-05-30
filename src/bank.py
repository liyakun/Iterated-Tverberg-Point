__author__ = 'ya kun'

import test
import data
import plot
import regression

class Bank:

    """filter data with unknown features && create training && testing data matrix"""
    my_data = data.Data()
    my_data.data_ready()
    print "Positive Instances Percentage is %f " % my_data.get_positive_instances_percent()

    def runbank(self):
        """ regression algorithm, my_data, num_of_train, num_of_training_data, num_of_iteration_each_train """
        weights, weights_all = regression.Regression().grad_ascent(self.my_data, 3, 1000, 300)

        """write trained weights to file"""
        self.my_data.write_to_csv_file("../resources/bank/output_weights.csv", weights)

        """plot weights convergence from regression"""
        plot.Plot().plot(weights_all)

        """testing phase"""
        test.Test().perform_test(1, self.my_data.test_matrix, self.my_data.test_class_list, weights)
