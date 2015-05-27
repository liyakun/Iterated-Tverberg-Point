"""
This file implement the iterated Tverberg points algorithm
"""
from itertools import compress
import numpy as np
import Optimization as Opt

class IteratedTverberg:

    def __init__(self):
        pass

    def center_point(self, points):
        opt = Opt.Optimization()
        points = np.asarray(points)
        n, d = points.shape

        """
        The loop terminates when a point is in the bucket B_z

        np.ceil(a) Return the ceiling of the input, element-wise.
        The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as |x|.
        For example:
                    a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
                    np.ceil(a) => array([-1., -1., -0.,  1.,  2.,  2.,  2.])
        """

        z = int(np.log10(np.ceil(n/(2*((d+1)**2)))))

        # Initialize empty stacks / buckets
        B = [[] for l in range(z+1)]

        # Push initial points with trivial proofs
        # Proofs consist of a factor and a hull
        for s in points:
            # TODO (one could copy the proof to be save)
            proof = [(1, s)]
            B[0].append((s, [proof]))

        while len(B[z]) == 0:
            # Initialize proof to be empty stack
            proof = []

            # Let l be the max such that B_l-1 has at least d+2 points
            # ToDO: optimize?
            l = self.find_l(B, d)

            # Pop d + 2 points q_1, ... , q_d+2 from B_l-1
            # qs denotes the list of points q_1, to q_d+2
            # qss denotes the collection of proofs for each point q_i
            qs_with_proof = opt.pop(B[l-1], d+2)
            qs, pss = zip(*qs_with_proof)

            # TODO: the proof parts should be "ordered" according to the paper
            # calculate the radon partition
            radon_pt, alphas, partition_masks = opt.randon_partition(qs)

            for k in range(2):
                # qs_part denotes the list of points in this partition
                # qs_part = list(compress(qs, partition_maks[k]))

                # pss_part denotes the collection of proofs for that points
                pss_part = list(compress(pss, partition_masks[k]))

                # as_part denotes the factors of the radon point in regard to
                # the hulls consisting of the partitions
                as_part = alphas[k]

                # Form a proof of depth 2^(l+1) for the radon point
                for i in range(2 ** (l-1)):

                    # union the i'th part of each proof of each point
                    X_alphas = []
                    X_hulls = []
                    for j, ps in enumerate(pss_part):
                        S_ij = ps[j]

                        for ppt in S_ij:
                            # Adjust the factors of the proofs to be able to
                            # describe the radon point as a combination of it's
                            # proofs
                            alpha = as_part[j] * ppt[0]
                            hull = ppt[1]

                            # Add them to the new proof
                            X_alphas.append(alpha)
                            X_hulls.append(hull)

                    # reduce the hull of the radon point, that is consisting
                    # of the proof parts, to d+1 hull points
                    X2, non_hull = opt.prune_zipped(X_alphas, X_hulls)

                    proof.append(X2)
                    B[0].extend(non_hull)
            B[l].append((radon_pt, proof))
        return B[z][0][0]



