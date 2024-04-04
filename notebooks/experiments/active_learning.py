import pickle as pkl
import warnings
from functools import partial

from modAL.models import ActiveLearner, Committee
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pymfe.mfe import MFE
import numpy as np
import pandas as pd
import openml

class ActiveLearningExperiment:
    def __init__(self, dataset_id, l_size, random_state=None):

        X, y = self.__load_data(dataset_id)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

        # Faz com que inicialmente haja uma instância rotulada por classe
        labeled_index = [
            np.random.RandomState(random_state).choice(np.where(y_train == cls)[0])
            for cls in np.unique(y_train)]

        if len(labeled_index) < l_size:

            additional_index = np.random.RandomState(random_state).choice(
                len(y_train), size=l_size, replace=False)

            labeled_index.extend(additional_index.tolist())

        self.labeled_index = labeled_index
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def run(self, estimator, query_strategy, n_queries=100, batch_size=5,
            committee_size=None):

        l_X_pool = self.X_train[self.labeled_index]
        l_y_pool = self.y_train[self.labeled_index]

        u_X_pool = np.delete(self.X_train, self.labeled_index, axis=0)
        u_y_pool = np.delete(self.y_train, self.labeled_index, axis=0)

        args = dict()
        args['estimator'] = estimator
        args['X_training'] = l_X_pool
        args['y_training'] = l_y_pool
        args['query_strategy'] = partial(query_strategy,
                                         n_instances=batch_size)

        if query_strategy.__module__ == 'modAL.uncertainty':
            learner = ActiveLearner(**args)
        else:
            learner_list = [ActiveLearner(**args) for _ in range(committee_size)]
            learner = Committee(learner_list=learner_list,
                                query_strategy=args['query_strategy'])

        scores = []
        
        for idx in range(n_queries):

            u_pool_size = np.size(u_y_pool)

            if u_pool_size <= 0:
                break

            query_index = (learner.query(u_X_pool)[0]
                           if u_pool_size > batch_size + 1
                           else np.arange(u_pool_size))

            learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])

            u_X_pool = np.delete(u_X_pool, query_index, axis=0)
            u_y_pool = np.delete(u_y_pool, query_index, axis=0)

            score = learner.score(self.X_test, self.y_test)
            scores.append(score)

        return scores

    def __load_data(self, dataset_id):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            dataset = openml.datasets.get_dataset(dataset_id)

        X, y, categorical_indicator, _ = dataset.get_data(
            target=dataset.default_target_attribute)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers = [('one-hot-encoder', encoder, categorical_indicator)]

        preprocessor = ColumnTransformer(transformers, remainder='passthrough')

        X, y = preprocessor.fit_transform(X), LabelEncoder().fit_transform(y)

        return X, y


from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from modAL.disagreement import (consensus_entropy_sampling,
                                max_disagreement_sampling,
                                vote_entropy_sampling)
from modAL.batch import uncertainty_batch_sampling

def meta_sampling(classifier, X_pool, n_instances):
    query_strategies = {
        'consensus_entropy_sampling': consensus_entropy_sampling,
        'entropy_sampling': entropy_sampling,
        'margin_sampling': margin_sampling,
        'max_disagreement_sampling': max_disagreement_sampling,
        'uncertainty_batch_sampling': uncertainty_batch_sampling,
        'uncertainty_sampling': uncertainty_sampling,
        'vote_entropy_sampling': vote_entropy_sampling
    }

    # carregando modelo
    with open('meta_model.pkl', 'rb') as f:
        model = pkl.load(f)

    # extração de metafeatures
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        mfe = MFE(groups='all')
        mfe.fit(X_pool)
        names, mfts = mfe.extract()

    X = [mfts]

    selected_strategy = model.predict(X)[0]

    print(selected_strategy)

    return query_strategies[selected_strategy](classifier, X_pool,
                                               n_instances=n_instances)

if __name__ == "__main__":

    from sklearn.svm import SVC
    from modAL.uncertainty import uncertainty_sampling
    from modAL.disagreement import max_disagreement_sampling

    exp = ActiveLearningExperiment(dataset_id=991,
                                   random_state=42,
                                   l_size=5)

    scores = exp.run(estimator=SVC(probability=True),
            committee_size=3,
            query_strategy=uncertainty_sampling)
    
    print(scores)
