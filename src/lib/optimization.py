# -*- coding: utf-8 -*-

from sklearn import linear_model
import numpy as np
import time

class Optimization:

    def __init__(self):
        pass

    # sigmoid function
    def sigmoid(self, inX):
        return 1.0/(1+np.exp(-inX))

    # gradient ascent optional optimize algorithm
    def grad_ascent(self, train_matrix, train_class_list, random_index_list_in_training, num_iteration, function):
        # get start time
        start_time = time.time()
        # get the number of rows and columns of the training matrix
        m, n = np.shape(train_matrix)
        # initialize a temporal weights vector to all 1
        weights_tmp = np.ones((n, 1))
        # iterate within the number of iteration
        for i in range(num_iteration):
            # get the index list of all the data in random_index_list_in_training
            data_index = range(len(random_index_list_in_training))
            for j in range(len(random_index_list_in_training)):
                # alpha changes on each iteration, improve oscillations
                # alpha decreases as the number of iterations increases, but it never reach 0
                alpha = 4/(1.0+j+i) + 0.001
                # randomly selecting each index from data_index, then get value from random_index_list_in_training
                # then we access the instance in train matrix to use in updating the weights, reduce periodic variations
                rand_index = int(np.random.uniform(0, len(data_index)))
                # get the sigmoid value
                h = function.sigmoid(np.sum(train_matrix[random_index_list_in_training[rand_index]]*weights_tmp))
                # compare the sigmoid value with teacher, and store the current error
                error_tmp = (train_class_list[random_index_list_in_training[rand_index]] - h)
                # update weights in current training example
                weights_tmp += alpha * train_matrix[random_index_list_in_training[rand_index]].transpose() * error_tmp
                # remove the used instances index
                del (data_index[rand_index])
        print "\nTraining finished within %fs!\n" % (time.time() - start_time)
        return weights_tmp

    def gradient_descent_random(self, train_matrix, train_class_list, random_index_list_in_training):
        clf = linear_model.SGDClassifier(loss="log", n_jobs=-1)
        clf.fit(np.asarray(train_matrix)[random_index_list_in_training], np.asarray(train_class_list)[random_index_list_in_training]
                )
        return clf.coef_[0]

    def gradient_descent_equal(self, train_matrix, train_class_list):
        clf = linear_model.SGDClassifier(loss="log", n_jobs=-1)
        clf.fit(train_matrix, train_class_list)
        return clf.coef_[0]

    # sigmoid test function
    def sig_test(self, instance, weights):
        value = np.dot(instance, weights)
        sig_value = self.sigmoid(value)

        if sig_value > 0.5:
            return 1.0
        else:
            return 0.0

    # return a vector of x, which satisfies 'M*x=0'
    def solve_homogeneous(self, equations):
        assert (isinstance(equations, np.ndarray)), "ndarray required"
        # Singular Value Decomposition.
        u, s, vh = np.linalg.svd(equations)
        # the last one is the solution for the sum of multiply with points to zero, as we add all one before
        return vh.T[:, -1]

    # find the alphas to solve the equation
    def find_alphas(self, points):
        _points = np.asarray(points)
        n, m = _points.shape
        # <editor-fold desc="Description">
        """
        numpy.vstack(tup): stack array in sequence vertically(row wise)
                Take a sequence of arrays and stack them vertically to make a single array.
                Rebuild arrays divided by vsplit
                For example: a = np.array([1, 2, 3])
                             b = np.array([2, 3, 4])
                            np.vstack((a,b)) => array([[1, 2, 3],
                                                       [2, 3, 4]])
        We use b_i := (a_i, 1), so we have (d+2) vectors in d+1 dimensional space, thus they are linearly
        dependent. That is, there exists α_1, ... , α_m not all zero such that sum(α_i * b_i)=0, i in [1, m]
        """
        # </editor-fold>
        equations = np.vstack((np.ones(n), _points.T))
        return self.solve_homogeneous(equations)

    # find a radon partition
    def radon_partition(self, points):
        # <editor-fold desc="Description">
        """
         points: (n, d)-array like
                 where n is the number of points and d is the dimension of the points
         Return the radon points, the factors for the partition I and the partition J
         and two masking arrays, representing the partitions in reference to the input array.
            (radon point),
            (alpha_I, alpha_J),
            (mask_I, mask_J)
        """
        # </editor-fold>
        points = np.asarray(points)
        n, d = points.shape
        assert (n >= d + 2), "Not enough points"

        # get the array of alphas, get the mask of alphas
        alphas = self.find_alphas(points)
        positive_idx = alphas > 0
        positive_alphas = alphas[positive_idx]
        positive_points = points[positive_idx]
        non_positive_idx = ~ positive_idx
        non_positive_alphas = alphas[non_positive_idx]

        # <editor-fold desc="Description">
        """
        The convex hull of I and J must intersect, because they both contain the
        point(vectors) in  both the convex hull of {a_i|i in I} and {a_j|j in J}

                p = sum{(α_i/α)a_i} = sum{-(α_j/α)*a_j}, i in I & j in J
        where

                α = sum{α_i} = -sum{α_j}, i in J & i in I,
        """
        # </editor-fold>
        sum_alphas = np.sum(positive_alphas)
        radon_pt_positive_alphas = positive_alphas / sum_alphas
        radon_pt_non_positive_alphas = non_positive_alphas / (-sum_alphas)
        radon_pt = np.dot(radon_pt_positive_alphas, positive_points) # get common vectors, radon point
        return (radon_pt,
                (radon_pt_positive_alphas, radon_pt_non_positive_alphas),
                (positive_idx, non_positive_idx))

    # yield n element from the list l, IndexError if len(l) < n, pop the index of point with proof within one bucket
    def pop(self, l, n):
        for i in range(n):
            yield l.pop()

    # <editor-fold desc="Description">
    """
    let l be the max such that B_(l-1) has at least d+2 points
    when the B_0 are popped (d+2) points many times, then there are less than (d+2) points left finally, then we come
    to B_1, and find_l make sure we will find the radon point and partition on each level (in bucket_i)
    """
    # </editor-fold>
    def find_l(self, buckets, d):
        l = None
        for i, b in enumerate(buckets):
            if len(b) >= d + 2:
                l = i

        assert (l is not None), "No bucket with d+2 points found"
        return l + 1

    # prune based on caratheodory's theorem
    def prune_zipped(self, alphas, hull):
        _alphas = np.asarray(alphas)
        _hull = np.asarray(hull)
        alphas, hull, non_hull = self.prune_recursive(_alphas, _hull, [])
        assert (alphas.shape[0] == hull.shape[0]), "Broken hull"
        non_hull = [(p, [[(1,p)]]) for p in non_hull]
        return zip(alphas, hull), non_hull

    # <editor-fold desc="Description">
    """
    prune recursively by referencing http://www.math.cornell.edu/~eranevo/homepage/ConvNote.pdf
    x = sum{α_i*x_i}, x_i in S, such that α_i > 0 ; sum(α_i) = 1 and |S| > d + 1
        here we have |S| = d + 2
    """
    # </editor-fold>
    def prune_recursive(self, alphas, hull, non_hull):
        # remove all coefficients that are already (close to) zero
        idx_nonzero = ~ np.isclose(alphas, np.zeros_like(alphas))  # alphas != 0
        alphas = alphas[idx_nonzero]

        # Add pruned points to the non hull (and thus back to bucket B_0)
        non_hull = non_hull + hull[~idx_nonzero].tolist()

        hull = hull[idx_nonzero]
        n, d = hull.shape

        # continue prune until d+1 hull points, then can't  be reduced any further
        if n <= d + 1:
            return alphas, hull, non_hull

        # Choose d + 2 hull points
        _hull = hull[:d + 2]
        _alphas = alphas[:d + 2]

        # <editor-fold desc="Description">
        """
        create linearly dependent vectors

        we choose d+2 points last step, denote as x_1, ... , x_(d+2)
        then we consider vectors {x_2-x_1, x_3-x_1, ... , x_(d+2)-x_1}, those vectors are linearly dependent
        therefore, there exists β_2, ... , β_(d+2) not all zero such that sum{(x_i-x_1)β_i}=0, i in [2, d+2]
        Then we solve the equation β * lindep = 0
        """
        # </editor-fold>
        linear_dependent = _hull[1:] - _hull[1]
        _betas = self.solve_homogeneous(linear_dependent.T)

        # Then we need to find β_1 = sum{-(β_i)}, i >= 2
        beta1 = np.negative(np.sum(_betas))
        betas = np.hstack((beta1, _betas))

        # <editor-fold desc="Description">
        """
        calculate the adjusted alphas and determine the minimum
        calculate the minimum fraction alpha / theta_i, in this case for each theta_i > 0

        we want to reduce the size of x, and we get the β_1, β_2, ... , β_(d+2) with sum to zero
        and sum(β_i * x_i) = 0, then we can represent x = sum{α_i * x_i} = sum{(α_i - λ*β_i)x_i} for all λ
        Thus, we can choose a λ such that α'_i = α_i - λ*β_i >= 0, and at least one such value is 0.
        Still, sum(α'_i) = sum(α_i) = 1, so we get another representation of x whose support has size smaller than |S|
        """
        # </editor-fold>
        idx_positive = betas > 0
        idx_nonzero = ~ np.isclose(betas, np.zeros_like(betas))  # betas != 0
        idx = idx_positive & idx_nonzero  # make sure betas are positive and nonzero, so  α'_i = α_i - λ*β_i >= 0
        lambdas = _alphas[idx] / betas[idx]
        lambda_min_idx = np.argmin(lambdas)


        # <editor-fold desc="Description">
        # adjust the alpha's of the original point
        # since _alphas is a view to the original data, the alphas array will like
        # be updated automatically
        # </editor-fold>
        _alphas[:] = _alphas - (lambdas[lambda_min_idx] * betas)

        # remove (filter) the pruned hull vector
        idx = np.arange(n) != lambda_min_idx   # get the index of points which are Not corresponding to the minimum λ
        hull = hull[idx]                       # get the representation of the pruned convex hull
        non_hull.append(hull[lambda_min_idx])  # add pruned points to the non hull (and thus back to bucket B_0)
        alphas = alphas[idx]                   # get the corresponding alphas with the convex hull points

        return self.prune_recursive(alphas, hull, non_hull)
