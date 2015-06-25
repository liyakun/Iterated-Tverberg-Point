"""
This file contains the main function
"""
from src.model import skin

""" run skin test """
my_skin = skin.Skin()
# number_of_training, number_of_training_instances, number_of_fold, percentage_of_training(e.g. 7) 70%
my_skin.run_skin_n_fold(1000, 2000, 2, 7)

""" run bank test """
# my_bank = bank.Bank()
# my_bank.runbank(1000, 2000)

""" test plot error """
# my_plot = plot.Plot()
# my_plot.plot_error("../resources/skin/result/error_random.txt")
# my_plot.plot_error("../resources/skin/result/error_equal.txt")