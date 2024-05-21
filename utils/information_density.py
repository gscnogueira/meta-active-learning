from sklearn.metrics.pairwise import pairwise_distances
from modAL.density import information_density
from modAL.uncertainty import classifier_margin
from modAL.utils.selection import multi_argmax

def density_weighted(learner, X):
    margin = classifier_margin(learner, X)
    id = information_density(X, 'euclidean')

    dw_margin = margin*id

    return dw_margin


def density_weighted_sampling(learner, X, n_instances=1):
    dw_margin = density_weighted(learner, X)

    return multi_argmax(dw_margin, n_instances=n_instances)


def training_utility(learner, X_unlabeled, X_labeled):

    dw = density_weighted(learner, X_unlabeled)

    similarity_mtx = 1/(1 + pairwise_distances(X_unlabeled, X_labeled))
    tu = similarity_mtx.mean(axis=1)

    return dw/tu


def training_utility_sampling(learner, X_unlabeled, X_labeled, n_instances=1):

    tu = training_utility(learner, X_unlabeled, X_labeled)

    return multi_argmax(tu, n_instances=n_instances)
