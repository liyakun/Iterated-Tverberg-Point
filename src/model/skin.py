__author__ = 'kun'

import numpy as np

from src.lib import data, sklearnlib, itertver


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
        weights, scores, mean_point = my_sklearn.train_and_test(200, self.X, self.Y, 0.3)
        self.my_data.write_to_csv_file("../resources/skin/output_weights_skin", weights)
        self.my_data.write_score_to_file("../resources/skin/scores", scores)

        """running iterated tverberg algorithm to compute the centerpoint"""
        my_itertver = itertver.IteratedTverberg()
        center_point_with_proof = my_itertver.center_point(weights)
        center_point = center_point_with_proof[0]
        proof_points = center_point_with_proof[1]

        """plot the points, coefficients, middle points """
        # my_plot = plot.Plot()
        # my_plot.plot3dpoints(weights, coefficients, mean_point)
        print center_point
        print proof_points
        return weights, center_point, mean_point


