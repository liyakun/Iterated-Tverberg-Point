"""
This file contains the main function
"""
from src.lib import test
from src.model import haberman
from src.model import bank

""" run haberman test """
# my_haber = haberman.Haberman()
# weights, coefficients, mean_point = my_haber.run_sklearn()
# my_test = test.Test()
# my_test.perform_test(my_haber.X, my_haber.Y, weights, coefficients, mean_point, "../resources/haberman/error.txt")

""" run skin test """
# my_skin = skin.Skin()
# my_test = test.Test()
# weights, coefficients, mean_point = my_skin.run_sklean()
# my_test.perform_test(my_skin.X, my_skin.Y, weights, coefficients, mean_point, "../resources/skin/error.txt")

""" run bank test """
my_bank = bank.Bank()
my_bank.runbank()