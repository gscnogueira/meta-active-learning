import os
import warnings
from functools import partial

from tqdm import tqdm
from modAL.models import ActiveLearner, Committee
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pymfe.mfe import MFE
import numpy as np
import pandas as pd
import openml

from modAL import uncertainty as u, batch as b
from modAL import disagreement as d

from information_density import training_utility_sampling
# Estratégias a serem utilizadas
# - [X] Uncertainty Sampling
# - [X] QBC
# - [X] Expected Error Reduction (Accuracy/Entropy)
# - [X] DW (Possível implementar com medida de densidade fornecida)
# - [ ] TU (Possível implementar com medida de densidade fornecida)


class ActiveLearningExperiment:

    MAX_NUMBER_OF_CLASSES = 5

    DISAGREEMENT_STRATEGIES = [d.max_disagreement_sampling,
                               d.vote_entropy_sampling,
                               d.consensus_entropy_sampling] 

    UNCERTAINTY_STRATEGIES = [u.uncertainty_sampling,
                              u.margin_sampling,
                              u.entropy_sampling,
                              b.uncertainty_batch_sampling]
    
    def __init__(self, dataset_id, initial_labeled_size, n_queries, batch_size,
                 committee_size=3, random_state=None):

        self.batch_size = batch_size
        self.committee_size = committee_size
        self.dataset_id = dataset_id
        self.n_queries = n_queries

        X, y = self.__load_data(dataset_id)

        # TODO: verificar se os splits são sempre os mesmos toda vez que a classe é instanciada
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)

        self.classes_ = np.unique(y)

        # Faz com que inicialmente haja uma instância rotulada por classe
        labeled_index = [
            np.random.RandomState(random_state).choice(np.where(y_train == cls)[0])
            for cls in np.unique(y_train)]

        # TODO: verificar se esse comportamento é viável, visto que
        # adiciona mais informação à configuração inicial
        if (n_classes := len(labeled_index)) < initial_labeled_size:

            possible_choices = [i for i in range(len(y_train))
                               if i not in labeled_index]

            additional_index = np.random.RandomState(random_state).choice(
                possible_choices,
                size=initial_labeled_size - n_classes,
                replace=False)

            labeled_index.extend(additional_index.tolist())

        self.labeled_index = labeled_index
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def run(self, estimator, query_strategy):

        l_X_pool = self.X_train[self.labeled_index]
        l_y_pool = self.y_train[self.labeled_index]

        u_X_pool = np.delete(self.X_train, self.labeled_index, axis=0)
        u_y_pool = np.delete(self.y_train, self.labeled_index, axis=0)

        args = dict()
        args['estimator'] = estimator
        args['X_training'] = l_X_pool
        args['y_training'] = l_y_pool
        args['query_strategy'] = partial(query_strategy,
                                         n_instances=self.batch_size)

        if query_strategy.__module__ == 'modAL.uncertainty':
            learner = ActiveLearner(**args)
        else:
            learner_list = [ActiveLearner(**args) for _ in range(self.committee_size)]
            learner = Committee(learner_list=learner_list,
                                query_strategy=args['query_strategy'])

        scores = []
        for idx in tqdm(range(self.n_queries), desc=query_strategy.__name__):

            u_pool_size = np.size(u_y_pool)

            if u_pool_size <= 0:
                break

            query_index = (learner.query(u_X_pool)[0]
                           if u_pool_size > self.batch_size + 2
                           else np.arange(u_pool_size))

            learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])

            u_X_pool = np.delete(u_X_pool, query_index, axis=0)
            u_y_pool = np.delete(u_y_pool, query_index, axis=0)

            y_pred = learner.predict(self.X_test)

            score = f1_score(self.y_test, y_pred, average='macro')
            scores.append(score)

        return scores

    def run_topline(self, estimator, query_strategies: list):
        return self.__run_marker(estimator, query_strategies,
                                 self._topline_query, name='perfect_meta_sampling')

    def run_baseline(self, estimator, query_strategies: list):
        return self.__run_marker(estimator, query_strategies,
                                 self.__baseline_query, name='random_meta_sampling')

    def run_meta_query(self, estimator, meta_model):
        return self.__run_marker(
            estimator=estimator,
            query_strategies=None,
            marker_query=partial(self.__meta_sample_query,
                                 meta_model=meta_model),
            name='meta_sampling')

    def _extract_mfs(self, X, y):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mfe = MFE(groups='all')
            mfe.fit(X,y)
            mf_names, mf_values = mfe.extract()

        mfs = pd.Series(data=mf_values, index=mf_names)

        return mfs

    def _extract_unsupervised_mfs(self, X):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mfe = MFE(groups='all')
            mfe.fit(X)
            mf_names, mf_values = mfe.extract()

        mfs = pd.Series(data=mf_values, index=mf_names)
        return mfs

    def _extract_clustering_mfs(self, X):

        clusterer = self.__get_best_cluster(X)

        y = clusterer.labels_

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mfe = MFE(groups='clustering')
            mfe.fit(X, y)
            mf_names, mf_values = mfe.extract()

        mfs = pd.Series(data=mf_values, index=mf_names)
        return mfs

    def __run_marker(self, estimator, query_strategies: list, marker_query, name=None):

        l_X_pool = self.X_train[self.labeled_index]
        l_y_pool = self.y_train[self.labeled_index]

        u_X_pool = np.delete(self.X_train, self.labeled_index, axis=0)
        u_y_pool = np.delete(self.y_train, self.labeled_index, axis=0)

        scores = []
        choices = []

        for idx in tqdm(range(self.n_queries), desc=name):

            u_pool_size = np.size(u_y_pool)

            if u_pool_size <= 0:
                break

            query_index, score, choice = marker_query(
                estimator=estimator,
                query_strategies=query_strategies,
                l_pool=(l_X_pool, l_y_pool),
                u_pool=(u_X_pool, u_y_pool),
                query_number=idx)

            scores.append(score)
            choices.append(choice)

            new_X, new_y = u_X_pool[query_index], u_y_pool[query_index]

            l_X_pool = np.append(l_X_pool, new_X, axis=0)
            l_y_pool = np.append(l_y_pool, new_y, axis=0)

            u_X_pool = np.delete(u_X_pool, query_index, axis=0)
            u_y_pool = np.delete(u_y_pool, query_index, axis=0)

        return scores, choices

    def __meta_sample_query(self, estimator,
                            l_pool, u_pool,
                            query_number,
                            meta_model,
                            **kwargs):

        query_strategy_dict = {
            'consensus_entropy_sampling': d.consensus_entropy_sampling,
            'entropy_sampling': u.entropy_sampling,
            'margin_sampling': u.margin_sampling,
            'max_disagreement_sampling': d.max_disagreement_sampling,
            'uncertainty_batch_sampling': b.uncertainty_batch_sampling,
            'uncertainty_sampling': u.uncertainty_sampling,
            'vote_entropy_sampling': d.vote_entropy_sampling
        }

        l_X_pool, l_y_pool = l_pool
        u_X_pool, u_y_pool = u_pool
        u_pool_size = np.size(u_y_pool)

        # Extração de metafeatures

        # query_number = pd.Series(data=[query_number], index=['query_number'])
        uns_mfs = self._extract_unsupervised_mfs(u_X_pool)
        clst_mfs = self._extract_clustering_mfs(u_X_pool)

        mfs = pd.concat([uns_mfs, clst_mfs])

        mfs.drop(labels='num_to_cat', inplace=True)

        X = [mfs.values]

        pred_strategy = meta_model.predict(X)[0]
        query_strategy = query_strategy_dict[pred_strategy]

        learner = self.__gen_learner(query_strategy=query_strategy,
                                     estimator=estimator,
                                     X_training=l_X_pool,
                                     y_training=l_y_pool)

        query_index = (learner.query(u_X_pool)[0]
                       if u_pool_size > self.batch_size + 2
                       else np.arange(u_pool_size))

        learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])

        y_pred = learner.predict(self.X_test)
        score = f1_score(self.y_test, y_pred, average='macro')

        return query_index, score, query_strategy.__name__

    def __baseline_query(self, estimator,
                         l_pool, u_pool,
                         query_strategies,
                         **kwargs):

        l_X_pool, l_y_pool = l_pool
        u_X_pool, u_y_pool = u_pool
        u_pool_size = np.size(u_y_pool)

        query_strategy = np.random.choice(query_strategies)

        learner = self.__gen_learner(query_strategy=query_strategy,
                                     estimator=estimator,
                                     X_training=l_X_pool,
                                     y_training=l_y_pool)

        query_index = (learner.query(u_X_pool)[0]
                       if u_pool_size > self.batch_size + 2
                       else np.arange(u_pool_size))

        learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])
        
        y_pred = learner.predict(self.X_test)
        score = f1_score(self.y_test, y_pred, average='macro')

        return query_index, score, query_strategy.__name__


    def _topline_query(self, estimator,
                       l_pool, u_pool,
                       query_strategies,
                       **kwargs):

        best_score = 0
        best_sample = None
        best_strategy = None
        u_X_pool, u_y_pool = u_pool
        l_X_pool, l_y_pool = l_pool

        u_pool_size = np.size(u_y_pool)

        for qs in query_strategies:

            # TODO: implementar condição de maneira menos grotesca
            learner = self.__gen_learner(
                estimator=estimator,
                X_training=l_X_pool,
                y_training=l_y_pool,
                query_strategy=(qs if qs != training_utility_sampling
                                else partial(qs, X_labeled=l_X_pool)))

            # TODO: Analisar a viabilidade dessa query final
            query_index = (learner.query(u_X_pool)[0]
                           if u_pool_size > self.batch_size + 2
                           else np.arange(u_pool_size))

            learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])

            y_pred = learner.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average='macro')

            if score > best_score:
                best_score = score
                best_sample = query_index
                best_strategy = qs

        return best_sample, best_score, best_strategy.__name__

    def __gen_learner(self, estimator, query_strategy, X_training, y_training):

        qs = partial(query_strategy, n_instances=self.batch_size)

        if query_strategy in self.DISAGREEMENT_STRATEGIES:

            learner_list = []

            for _ in range(self.committee_size):
                try:
                    learner = ActiveLearner(estimator=estimator,
                                            query_strategy=qs,
                                            X_training=X_training,
                                            y_training=y_training,
                                            bootstrap_init=True)
                    if not np.array_equal(self.classes_,
                                          learner.estimator.classes_):
                        # Não há um número de instâncias suficiente
                        # para o uso de bootstrap. Portanto,  o procedimento
                        # padrão será realizado
                        raise ValueError

                except ValueError:
                    learner = ActiveLearner(estimator=estimator,
                                            query_strategy=qs,
                                            X_training=X_training,
                                            y_training=y_training)

                learner_list.append(learner)

            learner = Committee(learner_list=learner_list,
                                query_strategy=qs)

            return learner

        else:

            learner = ActiveLearner(estimator=estimator,
                                    query_strategy=qs,
                                    X_training=X_training,
                                    y_training=y_training)
            return learner


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

    def __get_best_cluster(self, X):

        range_n_clusters = range(2, min(X.shape[0]+1,
                                        self.MAX_NUMBER_OF_CLASSES+1))

        # Inicia max_score com valor suficientemente pequeno
        max_score = -10
        best_clusterer = None

        for n_clusters in range_n_clusters:

            try:
                clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
                cluster_labels = clusterer.fit_predict(X)

                current_score = silhouette_score(X, cluster_labels)

            except ValueError:
                continue

            if current_score > max_score:
                max_score = current_score
                best_clusterer = clusterer

        return best_clusterer


