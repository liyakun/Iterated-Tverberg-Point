# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn import linear_model
from sklearn import cross_validation
import numpy as np
import Data
import IteratedTverberg

weigths, scores = [], []

my_haber_man_data = Data.Data()
my_haber_man_data.load_haber_man_data()
X, y = my_haber_man_data.parse_haber_man_data()  # get instances matrix and corresponding label

logistic_reg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
Y = np.array(y.transpose())[0]

skf = cross_validation.StratifiedShuffleSplit(Y, 1000, test_size=0.3, random_state=0)
for train_idx, test_idx in skf:
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    logistic_reg.fit(X_train, Y_train)
    weigths.append(logistic_reg.coef_[0])
    scores.append(logistic_reg.score(X_test, Y_test))

my_haber_man_data.write_to_csv_file("../resources/haberman/output_weights_haber_man", weigths)
my_haber_man_data.write_score_to_file("../resources/haberman/scores", scores)


my_itertverberg = IteratedTverberg.IteratedTverberg()
coefficients = my_itertverberg.center_point(weigths)
print coefficients