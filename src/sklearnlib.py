import numpy as np
from sklearn import linear_model
from sklearn import cross_validation

class Sklearnlib:

    def training(self, x_train, y_train):
        logistic_reg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
        logistic_reg.fit(x_train, y_train)

    def testing(self, x_test, y_test):
        logistic_reg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
        logistic_reg.score(x_test, y_test)

    def train_and_test(self, iteration, x, y, test_size):
        weights, scores = [], []
        logistic_reg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
        skf = cross_validation.StratifiedShuffleSplit(y, iteration, test_size, random_state=0)
        for train_idx, test_idx in skf:
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            logistic_reg.fit(x_train, y_train)
            weights.append(logistic_reg.coef_[0])
            scores.append(logistic_reg.score(x_test.astype(np.float), y_test.astype(np.float)))
        return weights, scores, self.get_mean_point(weights)

    def get_mean_point(self, weights):
        _x, _y, _z = [], [], []
        for i in range(len(weights)):
            _x.append(weights[i][0])
            _y.append(weights[i][1])
            _z.append(weights[i][2])
        mean_point = [np.mean(_x), np.mean(_y), np.mean(_z)]
        return mean_point