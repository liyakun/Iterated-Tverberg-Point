'''
Created on 16.10.2014
@author: mkamp
'''

from math import log
from math import sqrt
from random import random
from scipy.stats import bernoulli
from scipy.stats import rv_discrete
from scipy.stats import uniform

import sparse_vector


class InputStream(object):

    def __init__(self,identifier,nodes=1):
        self.generated_examples=0
        self.current_round=1
        self.number_of_nodes=nodes
        self._identifier=identifier

    def get_identifier(self):
        return self._identifier

    def set_identifier(self,identifier):
        self._identifier=identifier

    identifier=property(get_identifier,set_identifier,None,"identifier of InputStream that is, e.g., used in experiment logs")

    def generate_example(self):
        self.generated_examples+=1
        if self.generated_examples % self.number_of_nodes==0:
            if (self.current_round) % 1000 == 0:
                print "Completed round: "+str(self.current_round)
            self.current_round+=1
        return self._generate_example()

    def has_more_examples(self):
        return True

    def _generate_drift_event(self):
        print "drift: %d" % (self.current_round)

def draw_random_subset(S, p):
    res = set([])
    for s in S:
        if random() < p:
            res.add(s)
    return res

class UniformRecordGenerator():
    def __init__(self, number_of_items, inclusion_prob):
        self.singltetons = set(range(number_of_items))
        self.inclusion_prob = inclusion_prob

    def get_singletons(self):
        return self.singltetons

    def get_average_record_length(self):
        return len(self.singltetons) * self.inclusion_prob

    def generate_record(self):
        return draw_random_subset(self.singltetons, self.inclusion_prob)

class Disjunction():
    def __init__(self, singletons, literals=set([])):
        self.literals = literals
        self.singletons = singletons

    def get_singletons(self):
        return self.singletons

    def get_labels(self):
        return set([-1, 1])

    def get_label(self, record):
        if not(record.isdisjoint(self.literals)):
            return 1
        return -1

class SyntheticDataGenerator(InputStream):
    def __init__(self, record_generator, model_drifter, nodes = 1):
        self.record_generator = record_generator
        self.model_drifter = model_drifter
        self.model = model_drifter.construct_init_model()
        self.examples_per_round = nodes
        self.examples_in_current_round = 0
        InputStream.__init__(self, self.get_identifier(), nodes)

    def _generate_example(self):
        record = self.record_generator.generate_record()
        label = self.model.get_label(record)
        self.examples_in_current_round += 1
        if self.examples_in_current_round == self.examples_per_round:
            has_drifted = self.model_drifter.drift(self.model, self.get_identifier())
            if has_drifted: self._generate_drift_event()
            self.examples_in_current_round = 0
        return (sparse_vector.from_unit_cube(record), label)

    def get_identifier(self):
        return self.model_drifter.get_identifier() + "(" + str(len(self.model.singletons)) + ")"

class Drifter():
    def drift(self, model, stream_identifier):
        if random() < self.change_prob:
            self._do_drift(model)
            return True
        else:
            return False

class RapidDisjunctionDrifter(Drifter):
    def __init__(self, singletons, change_prob, inclusion_prob):
        self.singletons = singletons
        self.change_prob = change_prob
        self.inclusion_prob = inclusion_prob

    def get_identifier(self):
        return "RpDj(%d, %s)" % (len(self.singletons), str(self.change_prob))

    def construct_init_model(self):
        return Disjunction(self.singletons, draw_random_subset(self.singletons, self.inclusion_prob))

    def _do_drift(self, disjunction):
        disjunction.literals = draw_random_subset(disjunction.get_singletons(), self.inclusion_prob)

class RapidlyDriftingDisjunction(SyntheticDataGenerator):
    def __init__(self, number_of_items, change_prob, examples_per_round=1):
        self.item_inclusion_prob = sqrt(1 - pow(0.5, 1 / float(number_of_items)))
        record_generator = UniformRecordGenerator(number_of_items, self.item_inclusion_prob)
        model_drifter = RapidDisjunctionDrifter(record_generator.get_singletons(), change_prob, self.item_inclusion_prob)
        SyntheticDataGenerator.__init__(self, record_generator, model_drifter, examples_per_round)

class ZeroOneConditionedBernoulli(rv_discrete):
    def __init__(self, p0, p1):
        self.zeroBernoulli = bernoulli(p0)
        self.oneBernoulli = bernoulli(p1)

    def pmf(self, conditioner, value):
        if conditioner == 0:
            return self.zeroBernoulli.pmf(value)
        else:
            return self.oneBernoulli.pmf(value)

    def rvs(self, conditioner):
        if conditioner == 0:
            return self.zeroBernoulli.rvs()
        else:
            return self.oneBernoulli.rvs()

def next_hidden_assignment(assignment):
    i = 0
    while assignment[i] == 1:
        assignment[i] = 0
        i += 1
        if i == len(assignment):
            return False
    assignment[i] = 1
    return True

class BshoutyLongModel(InputStream):
    def __init__(self, dim, drift_prob):
        InputStream.__init__(self, "HiddenLayer" + "(" + str(dim) + ")",1)
        self.label_distribution = bernoulli(0.5)
        self.dim = dim
        self.drift_prob = drift_prob
        self.hidden_layer_size = int(log(dim, 2))
        self.set_random_parameters()
        self.examples_in_current_macro_round=0

    def set_random_parameters(self):
        self.hidden_layer = []
        for i in xrange(self.hidden_layer_size):
            relevance = uniform.rvs(loc=0.0, scale=1)
            p0 = uniform.rvs(loc=0, scale=1 - relevance)
            p1 = p0 + relevance
            if random()<0.5:
                swap=p0
                p0=p1
                p1=swap
            self.hidden_layer.append(ZeroOneConditionedBernoulli(p0, p1))
        self.output_layer = []
        self.effects = []
        for i in xrange(self.dim):
            relevance = uniform.rvs(loc=0.9, scale=0.1)
            p0 = uniform.rvs(loc=0, scale=1 - relevance)
            p1 = p0 + relevance
            if random()<0.5:
                swap=p0
                p0=p1
                p1=swap
            self.output_layer.append(ZeroOneConditionedBernoulli(p0, p1))
            self.effects.append(abs(p0 - p1))

    def compute_hidden_probs_given_label(self, hidden_assingment, label):
        res = 1
        for i in xrange(len(self.hidden_layer)):
            res *= self.hidden_layer[i].pmf(label, hidden_assingment[i])
        return res

    def compute_bayes_optimal_error(self):
        res = 0
        assignment = [0] * len(self.hidden_layer)
        new_assignment = True
        while new_assignment == True:
#            print assignment
#            print self.compute_hidden_probs_given_label(assignment, 1)
#            print self.compute_hidden_probs_given_label(assignment, 0)
            res += min(self.compute_hidden_probs_given_label(assignment, 1), self.compute_hidden_probs_given_label(assignment, 0)) / 2.0
            new_assignment = next_hidden_assignment(assignment)
        return res

    def get_labels(self):
        return [0, 1]

    def get_singletons(self):
        return range(self.dim)

    def get_identifier(self):
        return "HiddenLayer" + "(" + str(self.dim) + ")"

    def _generate_example(self):
        self._try_drift()
        label = self.label_distribution.rvs()
        hidden_values = []
        record = set()
        for i in xrange(len(self.hidden_layer)):
            hidden_values.append(self.hidden_layer[i].rvs(label))

        for i in xrange(self.dim):
            if self.output_layer[i].rvs(hidden_values[i % len(self.hidden_layer)]) == 1:
                record.add(i)
        if label == 0: label = -1
        record_vector = sparse_vector.SparseVector()
        for i in xrange(self.dim):
            if i in record:
                record_vector.components[i] = 1.0
            else:
                record_vector.components[i] = -1.0
        return (record_vector, label)

    def _try_drift(self):
        self.examples_in_current_macro_round+=1
        if self.examples_in_current_macro_round==self.number_of_nodes:
            self.examples_in_current_macro_round=0
            if uniform.rvs(loc=0.0, scale=1.0) < self.drift_prob:
                print "DRIFT!!!"
                self.set_random_parameters()
                self._generate_drift_event()