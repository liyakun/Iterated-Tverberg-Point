"""
This file contains the main function
"""
import Test
import Data
import Plot
import Regression

weights, weights_all = [], []

"""
filter data with unknown features && create training && testing data matrix
"""
my_data = Data.Data()
my_data.data_ready()

"""
training with regression algorithm
"""
weights, weights_all = Regression.Regression().grad_ascent(10, my_data.train_matrix, my_data.train_class_list, my_data.get_random_index_list(1500),
                                                           500)

"""
write trained weights to file
"""
my_data.write_to_csv_file("../resources/output_weights.csv", weights)

"""
plot weights convergence from regression
"""
Plot.Plot().plot(weights_all)

"""
testing phase
"""
Test.Test().perform_test(1, my_data.test_matrix, my_data.test_class_list, weights)
