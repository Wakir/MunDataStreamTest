"""Weighted Aging Ensemble."""

import numpy as np
import random
import math
from sklearn import base
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from ..ensembles.base import StreamingEnsemble
from ..ensembles import pruning

WEIGHT_CALCULATION = (
    "same_for_each",
    "proportional_to_accuracy",
    "kuncheva",
    "pta_related_to_whole",
    "bell_curve",
)

AGING_METHOD = ("weights_proportional", "constant", "gaussian")

class WAE(StreamingEnsemble):
    """
    Weighted Aging Ensemble.

    The method was inspired by Accuracy Weighted Ensemble (AWE) algorithm to which it introduces two main modifications: (I) classifier weights depend on the individual classifier accuracies and time they have been spending in the ensemble, (II) individual classifier are chosen on the basis on the non-pairwise diversity measure.

    :type base_estimator: ClassifierMixin class object
    :param base_estimator: Classification algorithm used as a base estimator.
    :type n_estimators: integer, optional (default=10)
    :param  n_estimators: The maximum number of estimators trained using consecutive data chunks and maintained in the ensemble.
    :type theta: float, optional (default=0.1)
    :param theta: Threshold for weight calculation method and aging procedure control.
    :type post_pruning: boolean, optional (default=False)
    :param post_pruning: Whether the pruning is conducted before or after adding the classifier.
    :type pruning_criterion: string, optional (default='accuracy')
    :param pruning_criterion: Selection of pruning criterion.
    :type weight_calculation_method: string, optional (default='kuncheva')
    :param weight_calculation_method: same_for_each, proportional_to_accuracy, kuncheva, pta_related_to_whole, bell_curve,
    :type aging_method: string, optional (default='weights_proportional')
    :param aging_method: weights_proportional, constant, gaussian.
    :type rejuvenation_power: float, optional (default=0.0)
    :param rejuvenation_power: Rejuvenation dynamics control of classifiers with high prediction accuracy.

    :vartype ensemble_: list of classifiers
    :var ensemble_: The collection of fitted sub-estimators.
    :vartype classes_: array-like, shape (n_classes, )
    :var classes_: The class labels.
    :vartype weights_: array-like, shape (n_estimators, )
    :var weights_: Classifier weights.

    :Examples:

    >>> import strlearn2 as sl
    >>> from sklearn.naive_bayes import GaussianNB
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.ensembles.WAE(GaussianNB())
    >>> ttt = sl.evaluators.TestThenTrain(
    >>> metrics=(sl.metrics.balanced_accuracy_score))
    >>> ttt.process(stream, clf)
    >>> print(ttt.scores)
    [[[0.91386218]
      [0.93032581]
      [0.90907219]
      [0.90544872]
      [0.90466186]
      [0.91956783]
      [0.90776942]
      [0.92685422]
      [0.92895186]
      ...
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        theta=0.1,
        post_pruning=False,
        pruning_criterion="accuracy",
        weight_calculation_method="kuncheva",
        aging_method="weights_proportional",
        rejuvenation_power=0.0,
        n_classifiers=3,
        scale=0.4,
        base_quality_measure=None,
        addOne = False,
        pruner_type=pruning.MultipleOffBestPruner
    ):
        """Initialization."""
        super().__init__(base_estimator, n_estimators, weighted=True)
        self.pruning_criterion = pruning_criterion
        self.theta = theta
        self.post_pruning = post_pruning
        self.weight_calculation_method = weight_calculation_method
        self.aging_method = aging_method
        self.rejuvenation_power = rejuvenation_power
        self.n_classifiers = n_classifiers
        self.scale = scale
        self.base_quality_measure = base_quality_measure
        self.addOne = addOne
        self.pruner_type = pruner_type

    def _prune(self):
        X, y = self.previous_X, self.previous_y
        pruner = self.pruner_type(
            self.ensemble_support_matrix(X), y, self.n_estimators, self.base_quality_measure
        )
        self._filter_ensemble(pruner.best_permutation)

    def _filter_ensemble(self, combination):
        self.ensemble_ = [self.ensemble_[clf_id] for clf_id in combination]
        self.iterations_ = self.iterations_[combination]
        self.weights_ = self.weights_[combination]

    def partial_fit(self, X, y, classes=None):
        # Initialization
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        #do przetestowania
        # self.previous_X, self.previous_y = (X, y)

        if len(self.ensemble_) == 0:
            self.weights_ = np.array([1])
            self.age_ = 0
            self.iterations_ = np.array([])

        # Scoring
        if self.age_ > 0:
            self.overall_accuracy = self.score(
                self.previous_X, self.previous_y)

        # Pre-pruning
        for x in range(self.n_classifiers):
            if len(self.ensemble_) > self.n_estimators and not self.post_pruning:
                self._prune()
            else:
                break

        # Preparing and training new candidate
        if self.addOne:
            candidate_clf = self._train_classifier(X, y)
            self.ensemble_.append(candidate_clf)
            self.iterations_ = np.append(self.iterations_, [1])
        else:
            for x in range(self.n_classifiers):
                candidate_clf = self._train_classifier(X, y)
                self.ensemble_.append(candidate_clf)
                self.iterations_ = np.append(self.iterations_, [1])

        self._set_weights()
        self._rejuvenate()
        self._aging()
        self._extinct()

        # Post-pruning
        for x in range(self.n_classifiers):
            if len(self.ensemble_) > self.n_estimators and self.post_pruning:
                self._prune()
            else:
                break

        # Weights normalization
        self.weights_ = self.weights_ / np.sum(self.weights_)

        # Ending procedure
        self.previous_X, self.previous_y = (X, y)
        self.age_ += 1
        self.iterations_ += 1

        return self
    
    def _train_classifier(self, X, y):
        irl = random.uniform(0.9, 1.1)
        # print("irl = " + str(irl))
        X_res, y_res = self.resample(X, y, irl, self.scale)
        candidate_clf = base.clone(self.base_estimator)
        candidate_clf.fit(X_res, y_res)
        return candidate_clf
    
    def resample(self, X, y, irl, scale):
        N = math.floor(X.shape[0] * scale)
        N_m = math.floor(N / (irl + 1))
        N_rm = N - N_m
        if np.shape(np.where(y==0))[0] < np.shape(np.where(y==1))[0]:
            minority_y = 0
            majority_y = 1
            minority_x = X[np.where(y==0)]
            majority_x = X[np.where(y==1)]
        else:
            minority_y = 1
            majority_y = 0
            minority_x = X[np.where(y==1)]
            majority_x = X[np.where(y==0)]
        bootstrap_sample = np.empty(shape=(N , majority_x.shape[1] + 1))
        for i in range(N_m):
            sample = minority_x[random.randint(0, minority_x.shape[0] - 1)]
            sample = np.insert(sample, 0 , minority_y)
            bootstrap_sample[i] = sample
        for i in range(N_rm):
            sample = majority_x[random.randint(0, majority_x.shape[0] - 1)]
            sample = np.insert(sample, 0 , majority_y)
            bootstrap_sample[N_m + i] = sample
        np.random.shuffle(bootstrap_sample)
        y_res = bootstrap_sample[:, 0]
        x_res = bootstrap_sample[:, 1:]
        return x_res, y_res

    def _accuracies(self):
        return np.array(
            [m_clf.score(self.previous_X, self.previous_y)
             for m_clf in self.ensemble_]
        )

    def _estimators_quality(self):
        if self.base_quality_measure is not None:
            return np.array(
                [self.base_quality_measure(self.previous_y, m_clf.predict(self.previous_X))
                 for m_clf in self.ensemble_]
            )
        else:
            return self._accuracies()

    def _set_weights(self):
        if self.age_ > 0:
            if self.weight_calculation_method == "same_for_each":
                self.weights_ = np.ones(len(self.ensemble_))

            elif self.weight_calculation_method == "kuncheva":
                accuracies = self._estimators_quality()
                self.weights_ = np.log(accuracies / (1.0000001 - accuracies))
                self.weights_[self.weights_ < 0] = 0

            elif self.weight_calculation_method == "pta_related_to_whole":
                accuracies = self._estimators_quality()
                self.weights_ = accuracies / self.overall_accuracy
                self.weights_[self.weights_ < self.theta] = 0

            elif self.weight_calculation_method == "bell_curve":
                accuracies = self._estimators_quality()
                self.weights_ = (
                    1.0
                    / (2.0 * np.pi)
                    * np.exp((self.overall_accuracy - accuracies) / 2.0)
                )
                self.weights_[self.weights_ < self.theta] = 0

        self.weights_ = np.nan_to_num(self.weights_)

    def _aging(self):
        if self.age_ > 0:
            if self.aging_method == "weights_proportional":
                # accuracies = self._estimators_quality()
                self.weights_ = self.weights_ / np.sqrt(self.iterations_)

            elif self.aging_method == "constant":
                self.weights_ -= self.theta * self.iterations_
                self.weights_[self.weights_ < self.theta] = 0

            elif self.aging_method == "gaussian":
                self.weights_ = (
                    1.0
                    / (2.0 * np.pi)
                    * np.exp((self.iterations_ * self.weights_) / 2.0)
                )

    def _rejuvenate(self):
        if self.rejuvenation_power > 0 and len(self.weights_) > 0:
            w = np.sum(self.weights_) / len(self.weights_)
            mask = self.weights_ > w
            self.iterations_[mask] -= self.rejuvenation_power * (
                self.weights_[mask] - w
            )

    def _extinct(self):
        combination = np.array(np.where(self.weights_ > 0))[0]
        if len(combination) > 0:
            self._filter_ensemble(combination)
