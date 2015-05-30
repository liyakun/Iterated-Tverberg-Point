"""
This file test weights w.r.t testing data
"""
import optimization
import numpy as np
class Test:

    errors_list = []

    def __init__(self):
        pass

    """ Perform test  """
    def perform_test(self, instance_matrix, labels_list, weights_list, coefficients, path):
        print "Test Starts...\n"
        my_optimization = optimization.Optimization()
        for j in range(len(weights_list)):
            print "%d th test..." % j
            error_count = 0
            for i in range(len(instance_matrix)):
                if int((my_optimization.sig_test(instance_matrix[i], np.asarray(weights_list[j])))) != int(labels_list[i]):
                    error_count += 1
            self.errors_list.append(float(error_count)/float(len(instance_matrix)))
            print "%d th test finished." % j

        error_c = 0
        for j in range(len(instance_matrix)):
            if int((my_optimization.sig_test(instance_matrix[i], np.asarray(coefficients)))) != int(labels_list[i]):
                    error_c += 1
        self.errors_list.append(float(error_c)/float(len(instance_matrix)))
        self.write_error_to_file(path)

    """ Write testing error rate to file """
    def write_error_to_file(self, path):
        with open(path, "w") as f:
            for i in range(0, len(self.errors_list)):
                f.writelines("Training Error of %dth weights vector: %f\n" % (i, self.errors_list[i]))
