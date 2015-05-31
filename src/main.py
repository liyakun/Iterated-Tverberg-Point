"""
This file contains the main function
"""
import skin
import test
import haberman

""" run haberman test """
my_haber = haberman.Haberman()
my_haber.run_sklearn()

""" run skin test """
# my_skin = skin.Skin()
# my_test = test.Test()
# weights, coefficients, mean_point = my_skin.run_sklean()
# my_test.perform_test(my_skin.X, my_skin.Y, weights, coefficients, mean_point, "../resources/skin/error.txt")
