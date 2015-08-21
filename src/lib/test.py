"""
This file test weights w.r.t testing data
"""
import numpy as np

from src.lib import optimization


class Test:

    def __init__(self):
        pass

    """ Perform test for all point """
    def perform_test_for_all_point(self, instance_matrix, labels_list, weights_all_list, path):
        errors_list = []
        print "Test Starts...\n"
        my_optimization = optimization.Optimization()
        for i in range(len(weights_all_list)):
            print "%dth test..." % i
            error_count = 0
            for j in range(len(instance_matrix)):
                if int((my_optimization.sig_test(np.transpose(np.asarray(instance_matrix[j])), np.asarray(weights_all_list[i][0])
                                                 ))) != int(labels_list[j]):
                    error_count += 1
            errors_list.append(float(error_count)/float(len(instance_matrix)))
            print "%d th test finished." % i
        self.write_error_to_file(errors_list, path)

    """ Perform test  """
    def perform_test(self, instance_matrix, labels_list, weights_list, coefficients, mean_point, weights_all, path):
        print "Test Starts...\n"
        my_optimization = optimization.Optimization()
        errors_list = []
        for j in range(len(weights_list)):
            print "%d th test..." % j
            error_count = 0
            for i in range(len(instance_matrix)):
                if int((my_optimization.sig_test((np.asarray(instance_matrix[i])), np.asarray(weights_list[j])
                                                 ))) != int(labels_list[i]):
                    error_count += 1
            errors_list.append(float(error_count)/float(len(instance_matrix)))
            print "%d th test finished." % j

        if coefficients is not None:
            error_center_counter, error_mean_counter, error_all_counter = 0, 0, 0
            print "test for middle and center point..."
            print mean_point, coefficients, weights_all[0]
            for i_ in range(len(instance_matrix)):
                if int((my_optimization.sig_test(instance_matrix[i_], np.asarray(mean_point).transpose()))) != \
                        int(labels_list[i_]):
                    error_mean_counter += 1
                if int((my_optimization.sig_test(instance_matrix[i_], np.asarray(coefficients).transpose()))) != \
                        int(labels_list[i_]):
                    error_center_counter += 1
                if int((my_optimization.sig_test(instance_matrix[i_], np.asarray(weights_all[0]).transpose()))) != \
                        int(labels_list[i_]):
                    error_all_counter += 1
            print "test for middle and center point finished."
            errors_list.append(float(error_mean_counter)/float(len(instance_matrix)))
            errors_list.append(float(error_center_counter)/float(len(instance_matrix)))
            errors_list.append(float(error_all_counter)/float(len(instance_matrix)))
        self.write_error_to_file(errors_list, path)

    """ Write testing error rate to file """
    def write_error_to_file(self, errors_list, path):
        with open(path, "w") as f:
            print len(errors_list)
            print errors_list
            for i in range(0, len(errors_list)):
                f.writelines("Testing Error of %dth weights vector: %f\n" % (i, errors_list[i]))
