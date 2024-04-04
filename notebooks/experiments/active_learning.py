import pickle as pkl
import warnings
from functools import partial

from tqdm import tqdm
from modAL.models import ActiveLearner, Committee
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pymfe.mfe import MFE
import numpy as np
import pandas as pd
import openml

from modAL import uncertainty as u, batch as b
from modAL import disagreement as d

class ActiveLearningExperiment:

    DISAGREEMENT_STRATEGIES = [d.max_disagreement_sampling,
                               d.vote_entropy_sampling,
                               d.consensus_entropy_sampling] 

    UNCERTAINTY_STRATEGIES = [u.uncertainty_sampling,
                              u.margin_sampling,
                              u.entropy_sampling,
                              b.uncertainty_batch_sampling]
    
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

    def run_topline(self, estimator, n_queries,
                    query_strategies: list,
                    batch_size=5, committee_size=3):

        l_X_pool = self.X_train[self.labeled_index]
        l_y_pool = self.y_train[self.labeled_index]

        u_X_pool = np.delete(self.X_train, self.labeled_index, axis=0)
        u_y_pool = np.delete(self.y_train, self.labeled_index, axis=0)

        scores = []

        for idx in tqdm(range(n_queries)):

            u_pool_size = np.size(u_y_pool)

            if u_pool_size <= 0:
                break

            query_index, score = self.__topline_query(
                estimator=estimator,
                query_strategies=query_strategies,
                l_pool=(l_X_pool, l_y_pool),
                u_pool=(u_X_pool, u_y_pool),
                batch_size=5,
                committee_size=committee_size)

            new_X, new_y = u_X_pool[query_index], u_y_pool[query_index]

            l_X_pool = np.append(l_X_pool, new_X, axis=0)
            l_y_pool = np.append(l_y_pool, new_y, axis=0)

            u_X_pool = np.delete(u_X_pool, query_index, axis=0)
            u_y_pool = np.delete(u_y_pool, query_index, axis=0)

            scores.append(score)

        return scores

    def __topline_query(self, estimator,
                        l_pool, u_pool,
                        query_strategies,
                        batch_size, committee_size):

        args = dict()
        args['estimator'] = estimator
        args['X_training'], args['y_training'] = l_pool

        active_learners = [self.__gen_learner(query_strategy=s,
                                              batch_size=batch_size,
                                              committee_size=committee_size,
                                              **args)
                           for s in query_strategies]

        best_score = 0
        best_sample = None
        u_X_pool, u_y_pool = u_pool

        u_pool_size = np.size(u_y_pool)

        for i, learner in enumerate(active_learners):

            query_index = (learner.query(u_X_pool)[0]
                           if u_pool_size > batch_size + 1
                           else np.arange(u_pool_size))

            learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])

            score = learner.score(self.X_test, self.y_test)

            if score > best_score:
                best_score = score
                best_sample = query_index

        return best_sample, best_score

    def __gen_learner(self, query_strategy, committee_size, batch_size, **kwargs):

        if query_strategy in self.UNCERTAINTY_STRATEGIES:
            query_strategy = partial(query_strategy, n_instances=batch_size)

            learner = ActiveLearner(query_strategy=query_strategy, **kwargs)

            return learner

        elif query_strategy in self.DISAGREEMENT_STRATEGIES:

            query_strategy = partial(query_strategy, n_instances=batch_size)

            learner_list = [ActiveLearner(**kwargs)
                            for _ in range(committee_size)]

            learner = Committee(learner_list=learner_list,
                                query_strategy=query_strategy)
            return learner

        else:
            raise ValueError("Estratégia de sampling não suportada")


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


if __name__ == "__main__":

    from sklearn.svm import SVC
    from modAL import uncertainty as u
    from modAL import disagreement as d
    from modAL import batch as b

    exp = ActiveLearningExperiment(dataset_id=991,
                                   random_state=42,
                                   l_size=5)

    query_strategies = [u.uncertainty_sampling,
                        u.margin_sampling,
                        u.entropy_sampling,
                        b.uncertainty_batch_sampling,
                        d.max_disagreement_sampling,
                        d.consensus_entropy_sampling,
                        d.vote_entropy_sampling]

    topline_scores = exp.run_topline(estimator=SVC(probability=True),
                                     query_strategies=query_strategies,
                                     n_queries=100,
                                     batch_size=5)
    print(topline_scores)
