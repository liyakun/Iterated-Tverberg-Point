"""
This file test weights w.r.t testing data
"""
import numpy as np

from src.lib import optimization


class Test:

    errors_list = []

    def __init__(self):
        pass

    """ Perform test  """
    def perform_test(self, instance_matrix, labels_list, weights_list, coefficients, mean_point, weights_all, path):
        print "Test Starts...\n"
        my_optimization = optimization.Optimization()
        for j in range(len(weights_list)):
            print "%d th test..." % j
            error_count = 0
            for i in range(len(instance_matrix)):
                if int((my_optimization.sig_test(np.transpose(np.asarray(instance_matrix[i])), np.asarray(weights_list[j])
                                                 ))) != int(labels_list[i]):
                    error_count += 1
            self.errors_list.append(float(error_count)/float(len(instance_matrix)))
            print "%d th test finished." % j

        if coefficients is not None:
            error_c, error_m, error_a = 0, 0, 0
            print "test for middle and center point..."
            for i in range(len(instance_matrix)):
                if int((my_optimization.sig_test(instance_matrix[i], np.asarray(mean_point).transpose()))) != \
                        int(labels_list[i]):
                    error_m += 1
                if int((my_optimization.sig_test(instance_matrix[i], np.asarray(coefficients).transpose()))) != \
                        int(labels_list[i]):
                    error_c += 1
                if int((my_optimization.sig_test(instance_matrix[i], np.asarray(weights_all).transpose()))) != \
                        int(labels_list[i]):
                    error_a += 1
            print "test for middle and center point finished."
            self.errors_list.append(float(error_m)/float(len(instance_matrix)))
            self.errors_list.append(float(error_c)/float(len(instance_matrix)))
            self.errors_list.append(float(error_a)/float(len(instance_matrix)))
        self.write_error_to_file(path)

    """ Write testing error rate to file """
    def write_error_to_file(self, path):
        with open(path, "w") as f:
            for i in range(0, len(self.errors_list)):
                f.writelines("Testing Error of %dth weights vector: %f\n" % (i, self.errors_list[i]))
