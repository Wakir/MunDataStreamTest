from builtins import range

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination.default import MaximumGenerationTermination

import numpy as np
import itertools
from sklearn import metrics

from strlearn2.ensembles.genetic_operators import FixedSizeSampling, BinaryMutation, BinaryCrossover, SubsetSelectionProblem

PRUNING_CRITERION = ('accuracy')

class OneOffPruner(object):
    def __init__(self, ensemble_support_matrix, y, ensemble_size, pruning_criterion='accuracy'):
        self.pruning_criterion = pruning_criterion
        self.ensemble_support_matrix = ensemble_support_matrix
        self.y = y
        self.ensemble_sie = ensemble_size
        self.best_permutation = self.optimise_ensemble()

    def optimise_ensemble(self):
        """
        Accuracy pruning.
        """
        candidates_no = self.ensemble_support_matrix.shape[0]

        loser = 0
        best_criterion = 0.
        for cid in range(candidates_no):
            weights = np.array(
                [0 if i == cid else 1 for i in range(candidates_no)])
            weighted_support = self.ensemble_support_matrix * \
                weights[:, np.newaxis, np.newaxis]
            acumulated_weighted_support = np.sum(weighted_support, axis=0)
            decisions = np.argmax(acumulated_weighted_support, axis=1)
            criterion = self.pruning_criterion(self.y, decisions)
            if criterion >  best_criterion:
                loser = cid
                best_criterion = criterion

        best_permutation = list(range(candidates_no))
        best_permutation.pop(loser)

        return best_permutation


class MultipleOffBestPruner(object):
    "Choose the best N classifiers"
    def __init__(self, ensemble_support_matrix, y, ensemble_size, pruning_criterion):
        self.ensemble_support_matrix = ensemble_support_matrix
        self.y = y
        self.ensemble_size = ensemble_size
        self.pruning_criterion = pruning_criterion
        self.best_permutation = self.optimise_ensemble()

    def optimise_ensemble(self):
        candidates_no = self.ensemble_support_matrix.shape[0]
        key = np.empty(candidates_no)
        # print(key.shape)
        for cid in range(candidates_no):
            weighted_support = self.ensemble_support_matrix[cid]
            # print("Weighted support:")
            # print(weighted_support.shape)
            decisions = np.argmax(weighted_support, axis=1)
            # print("Decisions:")
            # print(decisions.shape)
            key[cid] = self.pruning_criterion(self.y, decisions)

        # print(key)
        best_permutation = list(np.argpartition(key, -min(self.ensemble_size, len(key)))[-min(self.ensemble_size, len(key)):])
        # print("After")
        # print(key)
        # print(best_permutation)

        return best_permutation

        #problem = SubsetSelectionProblem(self.ensemble_support_matrix, self.y, self.ensemble_size,
        #                                 self.pruning_criterion)"""

class MultipleOffPruner(object):
    "Choose the best ensemble fo N classifiers"
    def __init__(self, ensemble_support_matrix, y, ensemble_size, pruning_criterion):
        self.ensemble_support_matrix = ensemble_support_matrix
        self.y = y
        self.ensemble_size = ensemble_size
        self.pruning_criterion = pruning_criterion
        self.best_permutation = self.optimise_ensemble()

    def optimise_ensemble(self):
        candidates_no = self.ensemble_support_matrix.shape[0]

        list_all = np.empty(candidates_no)
        for i in range(candidates_no):
            list_all[i] = i
        best_permutation = np.empty(self.ensemble_size)
        best_criterion = 0
        for subset in itertools.combinations(list_all, self.ensemble_size):
            weights = np.array(
                [1 if np.any(np.isin(subset, i)) else 0 for i in range(candidates_no)])
            weighted_support = self.ensemble_support_matrix * \
                weights[:, np.newaxis, np.newaxis]
            acumulated_weighted_support = np.sum(weighted_support, axis=0)
            decisions = np.argmax(acumulated_weighted_support, axis=1)
            criterion = self.pruning_criterion(self.y, decisions)
            if criterion > best_criterion:
                best_permutation = np.array(subset)
                best_criterion = criterion

        return list(best_permutation.astype(int))

class GeneticPruning(object):
    def __init__(self, ensemble_support_matrix, y, ensemble_size, pruning_criterion):
        self.ensemble_support_matrix = ensemble_support_matrix
        self.y = y
        self.ensemble_size = ensemble_size
        self.pruning_criterion = pruning_criterion
        self.best_permutation = self.optimise_ensemble()

    def optimise_ensemble(self):
        problem = SubsetSelectionProblem(self.ensemble_support_matrix, self.y, self.ensemble_size,
                                         self.pruning_criterion)
        algorithm = GA(
            pop_size=100,
            sampling=FixedSizeSampling(),
            crossover=BinaryCrossover(),
            mutation=BinaryMutation(),
            eliminate_duplicates=True
        )
        termination = MaximumGenerationTermination(100)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=False,
                       verbose=False)

        return np.where(res.X)[0]
