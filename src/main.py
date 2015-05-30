"""
This file contains the main function
"""
import haberman
import skin
import test

""" run haber man test """
# my_haber_man = haberman.Haberman()
# my_haber_man.run_sklearn()

""" run skin test """
my_skin = skin.Skin()
my_test = test.Test()
weights, coefficients = my_skin.run_sklean()
my_test.perform_test(my_skin.X, my_skin.Y, weights, coefficients, "../resources/skin/error.txt")
