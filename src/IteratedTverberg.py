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

        # np.ceil(a) return the ceiling of the input, element-wise, get the bigger integer than a, -1.7->-1, 0.2->1
        z = int(np.log10(np.ceil(n/(2*((d+1)**2)))))

        # initialize empty stacks / buckets with z+1 rows
        buckets = [[] for l in range(z+1)]

        # push initial points with trivial proofs with depth 1, proofs consist of a factor and a hull
        for s in points:
            buckets[0].append((s, [(1, s)]))

        # loop terminates when a point is in the bucket B_z
        while len(buckets[z]) == 0:

            # initialize proof to be empty stack
            proof = []

            # let l be the max such that B_(l-1) has at least d+2 points
            l = opt.find_l(buckets, d)

            """
            pop d + 2 points q_1, ... , q_d+2 from B_l-1,
            points_list denotes the list of points p_1, to p_(d+2),
            proofs_list denotes the collection of proofs for each point p_i
            zip(*iterables) : make an iterator that aggregates elements from each of the iterables.
            """
            points_list, proofs_list = zip(*opt.pop(buckets[l-1], d+2))

            # calculate the radon partition
            radon_pt, alphas, partition_idx_tuple = opt.randon_partition(points_list)

            """
            TODO: the proof parts should be "ordered" according to the paper

            Given a set of d+2 points with disjoint proofs of depth r, the Radon point of P has depth at least 2r.

            Let (P_1, P_2) be the Radon partition for P, and let c be the Radon point. For each p_i in P, order the
            parts in the proof partition of p_i
            """

            # given a set P of d+2 points with disjoint proofs of depth r, the Radon point of P has depth at least 2r.
            for k in range(2):
                """
                The ITERATEDTVERBERG algorithm is very similar to the ITERATEDRADON algorithm, the key difference
                is that each successive approximation carries with it a proof of its depth. When we combine d+2
                points of depth r into a Radon point c, we can rearrange the proofs to get a new proof that c has
                depth 2r
                """
                # ('ABCDEF', [1,0,1,0,1,1]) --> A C E F, compress(data, selector), collection of proofs for that points
                radon_pt_proof_list = list(compress(proofs_list, partition_idx_tuple[k]))

                # factors of the radon point in regard to the hulls consisting of the partitions
                radon_pt_factor_tuple = alphas[k]

                # form a proof of depth 2^(l+1) for the radon point, by lemma 4.1
                for i in range(2 ** (l-1)):

                    # union the i'th part of each proof of each point
                    pt_alphas, pt_hulls = [], []

                    # enumerate the proof list, get the index and proof
                    for j, ps in enumerate(radon_pt_proof_list):
                        S_ij = ps[i]    # get the i-th part of proof for point j

                        for ppt in S_ij:
                            # Adjust the factors of the proofs to be able to
                            # describe the radon point as a combination of it's
                            # proofs
                            alpha = radon_pt_factor_tuple[j] * ppt[0]
                            hull = ppt[1]

                            # Add them to the new proof
                            pt_alphas.append(alpha)
                            pt_hulls.append(hull)

                    # reduce the hull of the radon point, that is consisting
                    # of the proof parts, to d+1 hull points
                    X2, non_hull = opt.prune_zipped(pt_alphas, pt_hulls)

                    proof.append(X2)
                    buckets[0].extend(non_hull)
            buckets[l].append((radon_pt, proof))
        return buckets[z][0][0]