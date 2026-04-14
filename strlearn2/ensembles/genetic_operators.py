from pymoo.core.problem import ElementwiseProblem

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

import numpy as np


class FixedSizeSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class BinaryMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False

        return X


class SubsetSelectionProblem(ElementwiseProblem):
    def __init__(self, ensemble_support_matrix, y, n_max, criterion):
        self.ensemble_support_matrix = ensemble_support_matrix
        self.y = y
        self.n_max = n_max
        self.criterion = criterion
        super().__init__(n_var=len(self.ensemble_support_matrix),
                         n_obj=1,
                         n_constr=1,
                         xl=np.full((len(self.ensemble_support_matrix),), 0),
                         xu=np.full((len(self.ensemble_support_matrix),), 1),
                         type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        chosen_ensemble_support = self.ensemble_support_matrix[x == 1]
        acumulated_weighted_support = np.sum(chosen_ensemble_support, axis=0)
        decisions = np.argmax(acumulated_weighted_support, axis=1)
        criterion = self.criterion(self.y, decisions)

        out["F"] = -criterion
        out["G"] = (self.n_max - np.sum(x)) ** 2

