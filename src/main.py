"""
This file contains the main function
"""
import test
import data
import plot
import regression

weights, weights_all = [], []

"""
filter data with unknown features && create training && testing data matrix
"""
my_data = data.Data()
my_data.data_ready()
print "Positive Instances Percentage is %f " % my_data.get_positive_instances_percent()

"""
training with regression algorithm, my_data, num_of_train, num_of_training_data, num_of_iteration_each_train
"""
weights, weights_all = regression.Regression().grad_ascent(my_data, 3, 1000, 300)

"""
write trained weights to file
"""
my_data.write_to_csv_file("../resources/output_weights.csv", weights)

"""
plot weights convergence from regression
"""
plot.Plot().plot(weights_all)

"""
testing phase
"""
test.Test().perform_test(1, my_data.test_matrix, my_data.test_class_list, weights)
