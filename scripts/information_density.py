from modAL.density import information_density
from modAL.uncertainty import classifier_margin
from modAL.utils.selection import multi_argmax

def density_weighted(learner, X, n_instances=1):
    margin = classifier_margin(learner, X)
    id = information_density(X, 'euclidean')

    dw_margin = margin*id
    
    return multi_argmax(dw_margin, n_instances=n_instances)

