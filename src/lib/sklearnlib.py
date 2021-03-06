import numpy as np
from sklearn import linear_model, cross_validation, datasets

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

    def split_train_and_test(self, x_, y_, train_size):
        return cross_validation.train_test_split(x_, y_, train_size=train_size, random_state=42)

    def generate_fake_data(self, n_samples, n_features, n_informative, n_classes):
        X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                            n_redundant=0, n_classes=n_classes, random_state=42)
        return X, y