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
