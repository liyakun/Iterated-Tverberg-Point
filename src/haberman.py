# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
import numpy as np
import data
import itertver
import plot
import sklearnlib

class Haberman:

    my_haber_man_data = data.Data()
    my_haber_man_data.load_haber_man_data()
    X, y = my_haber_man_data.parse_haber_man_data()  # get instances matrix and corresponding label
    Y = np.array(y.transpose())[0]

    def run_sklearn(self):
        my_sklearn = sklearnlib.Sklearnlib()
        weights, scores = my_sklearn.traingandtest(100, self.X, self.Y, 0.3)
        self.my_haber_man_data.write_to_csv_file("../resources/haberman/output_weights_haber_man", weights)
        self.my_haber_man_data.write_score_to_file("../resources/haberman/scores", scores)

        my_itertver = itertver.IteratedTverberg()
        coefficients = my_itertver.center_point(weights)

        my_plot = plot.Plot()
        my_plot.plot3dpoints(weights, coefficients)
        print coefficients
