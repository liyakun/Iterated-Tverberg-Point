__author__ = 'kun'

import data
import plot
import optimization
import itertver
import numpy as np
import sklearnlib

class Skin:

    """prepare data"""
    my_data = data.Data()
    my_data.load_skin_data()
    skin_data = np.asarray(my_data.skin_data)
    np.random.shuffle(skin_data)
    X, Y = np.hsplit(skin_data, np.array([3, ]))
    Y = Y.transpose()[0]
    Y[Y == '2'] = 0
    X = X.astype(np.float)
    Y = Y.astype(np.float)

    def run_sklean(self):

        """training and testing"""
        my_sklearn = sklearnlib.Sklearnlib()
        weights, scores, mean_point = my_sklearn.train_and_test(100, self.X, self.Y, 0.2)
        self.my_data.write_to_csv_file("../resources/skin/output_weights_skin", weights)
        self.my_data.write_score_to_file("../resources/skin/scores", scores)

        """running iterated tverberg algorithm to compute the centerpoint"""
        my_itertver = itertver.IteratedTverberg()
        coefficients = my_itertver.center_point(weights)

        """plot the points, coefficients, middle points """
        my_plot = plot.Plot()
        my_plot.plot3dpoints(weights, coefficients, mean_point)

        return weights, coefficients, mean_point


