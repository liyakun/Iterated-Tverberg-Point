"""
This file test weights w.r.t testing data
"""
import optimization
class Test:

    errors_list = []

    def __init__(self):
        pass

    """
    Perform test
    """
    def perform_test(self, times, instance_matrix, labels_list, weights_list):
        print "Test Starts...\n"
        my_optimization = optimization.Optimization()
        for j in range(len(weights_list)):
            print "%d th test..." % j
            error_count = 0
            for i in range(len(instance_matrix)):
                if int((my_optimization.sig_test(instance_matrix[i], weights_list[j]))) != int(labels_list[i]):
                    error_count += 1
            self.errors_list.append(float(error_count)/float(len(instance_matrix)))
            print "%d th test finished." % j
        self.write_error_to_file()

    """
    Write testing error rate to file
    """
    def write_error_to_file(self):
        with open("../resources/bank/error.txt", "w") as f:
            for i in range(0, len(self.errors_list)):
                f.writelines("Training Error of %dth weights vector: %f\n" % (i, self.errors_list[i]))
