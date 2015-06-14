__author__ = 'ya kun'

from src.lib import data, test, plot, regression
from src.lib import itertver
import numpy as np

class Bank:

    """filter data with unknown features && create training && testing data matrix"""
    my_data = data.Data()
    my_data.data_ready()
    print "Positive Instances Percentage is %f " % my_data.get_positive_instances_percent()

    def runbank(self):
        """ regression algorithm, my_data, num_of_train, num_of_training_data, num_of_iteration_each_train """
        weights, weights_all = regression.Regression().grad_ascent(self.my_data, 1000, 2000, 300)

        """write trained weights to file"""
        self.my_data.write_to_csv_file("../resources/bank/output_weights.csv", weights)

        """plot weights convergence from regression"""
        # plot.Plot().plot(weights_all)

        # """find center point"""
        my_iterver = itertver.IteratedTverberg()
        center_point_with_proof = my_iterver.center_point(weights)
        print "Center Point with proof: ", center_point_with_proof[0]
        print "Center point: ", center_point_with_proof[0][0]
        print "Proof of center point: ", center_point_with_proof[0][1]
        print "Depth of center point: ", len(center_point_with_proof[0][1])
        #
        # """testing phase"""
        test.Test().perform_test(self.my_data.test_matrix, self.my_data.test_class_list, weights,
                         center_point_with_proof[0][0], center_point_with_proof[0][0], "../resources/bank/error.txt")
