"""
This file contains the main function
"""
from src.model import skin
from src.lib import plot
from src.model import fakedata

""" run skin test """
# my_skin = skin.Skin()
# number_of_training, number_of_training_instances, number_of_fold, percentage_of_training(e.g. 7) 70%,
# number of equally split disjoint subset
# we have many single vectors for the equal split, when we want to tes the number of single vector's influence,
# we can select subset from all the vectors calculated from the equal split
# my_skin.run_skin_n_fold(1000, 100, 10, 9, 2000)

""" run bank test """
# my_bank = bank.Bank()
# my_bank.runbank(1000, 2000)

""" test plot error """
# my_plot = plot.Plot()
# my_plot.plot_error("../resources/skin/result/error_random.txt")
# my_plot.plot_error("../resources/skin/result/error_equal.txt")


""" box plot """
my_plot = plot.Plot()

# number of files, and path of files
my_plot.box_plot_different_dimensions(10, "../resources/fakedata/result/dimension_differences_1000_single_vectors/")

""" test fake data """
# my_fake_data = fakedata.FakeData()
# n_samples, n_features, n_informative, n_classes
# percent_of_train(0~1), number_of_training, number_of_training_instances, number_of_experiments
# my_fake_data.run_fake_data_n_fold(10000, 9, 9, 2, 0.75, 3500, 1000, 10)
