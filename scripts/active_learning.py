import os
import warnings
from functools import partial
from collections import namedtuple

from modAL.models import ActiveLearner
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pymfe.mfe import MFE
import numpy as np
import pandas as pd
import openml


from information_density import training_utility_sampling
import config


class ActiveLearningExperiment:

    MAX_NUMBER_OF_CLASSES = 5

    def __init__(self, dataset_id, initial_labeled_size, n_queries, batch_size,
                 committee_size=3, random_state=None):

        self.batch_size = batch_size
        self.committee_size = committee_size
        self.dataset_id = dataset_id
        self.n_queries = n_queries

        X, y = self.__load_data(dataset_id)

        # TODO: verificar se os splits são sempre os mesmos toda vez
        # que a classe é instanciada

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
        if query_strategy == training_utility_sampling:
            return self.run_training_utility(estimator)[0]

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

        learner = ActiveLearner(**args)

        scores = []
        for idx in range(self.n_queries):

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

    def run_meta_query(self, estimator, meta_X, meta_y):
        return self.__run_marker(
            estimator=estimator,
            query_strategies=None,
            marker_query=partial(self.__meta_sample_query,
                                 meta_X=meta_X,
                                 meta_y=meta_y),
            name='meta_sampling')

    def run_training_utility(self, estimator):
        return self.__run_marker(
            estimator, [training_utility_sampling],
            self._topline_query, name='training_utility_sampling')

    def _extract_mfs(self, X, y):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mfe = MFE(groups='all')
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

        for idx in range(self.n_queries):

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

        Result = namedtuple('Result', 'scores choices' )
        return Result(scores, choices)

    def __gen_meta_model(self, X_train, y_train):

        meta_model = Pipeline([
            ('mean-inputer', SimpleImputer(missing_values=np.nan,
                                           strategy='mean')),
            ('meta-model', RandomForestClassifier())
        ])

        meta_model.fit(X_train.values, y_train)

        return meta_model

    def __meta_sample_query(self, estimator,
                            l_pool, u_pool,
                            query_number,
                            meta_X, meta_y,
                            **kwargs):

        l_X_pool, l_y_pool = l_pool
        u_X_pool, u_y_pool = u_pool
        u_pool_size = np.size(u_y_pool)

        print(meta_X.shape)
        print(meta_y.shape)

        print(f'Realizando query {query_number}')

        # Geração de meta-modelo
        meta_model = self.__gen_meta_model(meta_X, meta_y)

        # Extração de metafeatures
        pymfe_mfs = self._extract_mfs(l_X_pool, l_y_pool)
        query_number = pd.Series(data=[query_number], index=['query_number'])

        mfs = pd.concat([query_number, pymfe_mfs])
        mfs.drop(labels='num_to_cat', inplace=True)

        mfs.replace([np.inf, -np.inf], np.nan, inplace=True)

        X = [mfs.values]

        pred_strategy = meta_model.predict(X)[0]

        *_, perfect_local_choice = self._topline_query(
            estimator=estimator,
            l_pool=l_pool,
            u_pool=u_pool,
            query_strategies=config.query_strategies)

        query_strategy = config.query_strategy_dict[pred_strategy]

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

        meta_X.loc[len(meta_X)] = mfs
        meta_y[len(meta_y)] = perfect_local_choice

        Choice = namedtuple('Choice', 'pred true')
        return query_index, score, Choice(pred_strategy, perfect_local_choice)

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
                query_strategy=qs)

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

        if query_strategy == training_utility_sampling:
            qs = partial(query_strategy, n_instances=self.batch_size,
                         X_labeled=X_training)
        else:
            qs = partial(query_strategy, n_instances=self.batch_size)

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
