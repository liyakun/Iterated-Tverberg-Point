"""
This file implement the iterated Tverberg points algorithm
"""
from itertools import compress
import numpy as np
import optimization as Opt

class IteratedTverberg:

    def __init__(self):
        pass

    def center_point(self, points):
        opt = Opt.Optimization()
        points = np.asarray(points)
        n, d = points.shape
        # np.ceil(a) return the ceiling of the input, element-wise, get the bigger integer than a, -1.7->-1, 0.2->1
        z = int(np.log10(np.ceil(n/(2*((d+1)**2)))))
        # initialize empty stacks / buckets with z+1 rows
        buckets = [[] for l in range(z+1)]
        # push initial points with trivial proofs with depth 1, proofs consist of a factor and a hull
        for s in points:
            buckets[0].append((s, [[(1, s)]]))
        # loop terminates when a point is in the bucket B_z
        while len(buckets[z]) == 0:
            # initialize proof to be empty stack
            proof = []
            # let l be the max such that B_(l-1) has at least d+2 points
            l = opt.find_l(buckets, d)
            # <editor-fold desc="Description">
            """
            pop d + 2 points q_1, ... , q_d+2 from B_l-1,
            points_list denotes the list of points p_1, to p_(d+2),
            proofs_list denotes the collection of proofs for each point p_i
            zip(*iterables) : make an iterator that aggregates elements from each of the iterables.
            """
            # </editor-fold>
            idx = opt.pop(buckets[l-1], d+2)
            points_list, proofs_list = zip(*idx)
            # calculate the radon partition
            radon_pt, alphas, partition_idx_tuple = opt.radon_partition(points_list)
            # <editor-fold desc="Description">
            """
            TODO: the proof parts should be "ordered" according to the paper

            Given a set of d+2 points with disjoint proofs of depth r, the Radon point of P has depth at least 2r.

            Let (P_1, P_2) be the Radon partition for P, and let c be the Radon point. For each p_i in P, order the
            parts in the proof partition of p_i
            """
            # </editor-fold>
            for k in range(2):
                # <editor-fold desc="Description">
                """
                The ITERATEDTVERBERG algorithm is very similar to the ITERATEDRADON algorithm, the key difference
                is that each successive approximation carries with it a proof of its depth. When we combine d+2
                points of depth r into a Radon point c, we can rearrange the proofs to get a new proof that c has
                depth 2r

                # ('ABCDEF', [1,0,1,0,1,1]) --> A C E F, compress(data, selector), collection of proofs for that points
                """
                # </editor-fold>
                radon_pt_proof_list = list(compress(proofs_list, partition_idx_tuple[k]))
                # factors of the radon point in regard to the hulls consisting of the partitions
                radon_pt_factor_tuple = alphas[k]
                # <editor-fold desc="Description">
                """
                form a proof of depth 2^(l+1) for the radon point, by lemma 4.1
                In the case of the fig-2 of paper, the i is range(2), as next step we will get the radon point
                of depth 4, so l=1, then we get the depth of the four points which form the second order radon
                partition, the use of i here is just get the depth of the previous point, and depth is also the
                number of parts of the disjoint partitions of the proof of the previous point, so later we use
                ps[i] indicate the i part of the partitions of the proof of previous radon point
                """
                # </editor-fold>
                for i in range(2 ** (l-1)):
                    pt_alphas, pt_hulls = [], []
                    # enumerate the proof list, get the index and proof, i is the number of partitions in each proof
                    for j, ps in enumerate(radon_pt_proof_list):
                        # <editor-fold desc="Description">
                        """
                        get the i-th part of proof of point j
                        In the example of the paper, this is the second order point in the positive/negative
                        partition in radon_pt_proof_list, which contains two points(p_left, p_right) in this case.

                        Then we want to get the i-th part of the proof of p_left, which is basically the two lines which
                        form the proof of p_left

                        the format of {p_left, {p1, p2}, {p3, p4}}, the last two disjoint subsets form the proof, the
                        p_left has depth 2, as r=2 in this case
                        """
                        # </editor-fold>
                        parts_i_of_proof_of_j = ps[i]
                        for ppt in parts_i_of_proof_of_j:
                            # <editor-fold desc="Description">
                            """
                            Adjust the factors of the proofs to be able to describe the radon point as a combination
                            of it's proofs
                            """
                            # </editor-fold>
                            alpha = radon_pt_factor_tuple[j] * ppt[0]
                            hull = ppt[1]
                            # Add them to the new proof
                            pt_alphas.append(alpha)
                            pt_hulls.append(hull)
                    # reduce the hull of the radon point, that is consisting of the proof parts, to d+1 hull points
                    X2, non_hull = opt.prune_zipped(pt_alphas, pt_hulls)
                    proof.append(X2)
                    buckets[0].extend(non_hull)
            buckets[l].append((radon_pt, proof))
        return buckets[z][0][0]
