#! /usr/bin/pythonw
#This file test weights
import Optimization
class Test:

    errors_list = []

    def __init__(self):
        pass

    def performTest(self, times, instance_matrix, labels_list, weights_list):
        print "Test Starts...\n"
        myOptimization = Optimization.Optimization()
        for j in range(len(weights_list)):
            print "%d th test..." % j
            error_count = 0
            for i in range(len(instance_matrix)):
                if int((myOptimization.sigTest(instance_matrix[i], weights_list[j]))) != int(labels_list[i]):
                    error_count += 1
            self.errors_list.append(float(error_count)/float(len(instance_matrix)))
            print "%d th test finished." % j
        self.writeErrorToFile()

    def writeErrorToFile(self):
        with open("../resources/error.txt", "w") as f:
            for i in range(0, len(self.errors_list)):
                f.writelines("Training Error of %dth weights vector: %f" % (i, self.errors_list[i]))
