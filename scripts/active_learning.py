import pickle as pkl
import warnings
from functools import partial
from itertools import product

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

        self.classes_ = np.unique(y)

        # Faz com que inicialmente haja uma instância rotulada por classe
        labeled_index = [
            np.random.RandomState(random_state).choice(np.where(y_train == cls)[0])
            for cls in np.unique(y_train)]

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
                                 meta_model=meta_model))

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

        query_number = pd.Series(data=[query_number], index=['query_number'])
        uns_mfs = self._extract_unsupervised_mfs(u_X_pool)
        clst_mfs = self._extract_clustering_mfs(u_X_pool)

        mfs = pd.concat([query_number, uns_mfs, clst_mfs])

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

        args = dict()
        args['estimator'] = estimator
        args['X_training'], args['y_training'] = l_pool

        active_learners = [self.__gen_learner(query_strategy=s, **args)
                           for s in query_strategies]

        best_score = 0
        best_sample = None
        best_strategy = None
        u_X_pool, u_y_pool = u_pool

        u_pool_size = np.size(u_y_pool)

        for i, learner in enumerate(active_learners):

            query_index = (learner.query(u_X_pool)[0]
                           if u_pool_size > self.batch_size + 2
                           else np.arange(u_pool_size))

            learner.teach(X=u_X_pool[query_index], y=u_y_pool[query_index])


            y_pred = learner.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average='macro')

            if score > best_score:
                best_score = score
                best_sample = query_index
                best_strategy = query_strategies[i]

        return best_sample, best_score, best_strategy.__name__

    def __gen_learner(self, query_strategy, **kwargs):

        qs = partial(query_strategy, n_instances=self.batch_size)

        if query_strategy in self.UNCERTAINTY_STRATEGIES:

            learner = ActiveLearner(query_strategy=qs, **kwargs)
            return learner

        elif query_strategy in self.DISAGREEMENT_STRATEGIES:

            learner_list = []

            for _ in range(self.committee_size):
                try:
                    learner = ActiveLearner(bootstrap_init=True, **kwargs)
                    if not np.array_equal(self.classes_,
                                          learner.estimator.classes_):
                        # Não há um número de instâncias suficiente
                        # para o uso de bootstrap. Portanto,  o procedimento
                        # padrão será realizado
                        raise ValueError

                except ValueError:
                    learner = ActiveLearner(**kwargs)

                learner_list.append(learner)

            learner = Committee(learner_list=learner_list,
                                query_strategy=qs)

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

    def __get_best_cluster(self, X):

        range_n_clusters = range(2, min(X.shape[0]+1,
                                        self.MAX_NUMBER_OF_CLASSES+1))

        # Inicia max_score com valor suficientemente pequeno
        max_score = -10
        best_clusterer = None

        for n_clusters in range_n_clusters:

            clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
            cluster_labels = clusterer.fit_predict(X)

            current_score = silhouette_score(X, cluster_labels)

            if current_score > max_score:
                max_score = current_score
                best_clusterer = clusterer

        return best_clusterer

class MetaBaseBuilder(ActiveLearningExperiment):


    def run(self, estimator, n_queries,
            query_strategies: list,
            batch_size=5, committee_size=3):
        
        l_X_pool = self.X_train[self.labeled_index]
        l_y_pool = self.y_train[self.labeled_index]

        u_X_pool = np.delete(self.X_train, self.labeled_index, axis=0)
        u_y_pool = np.delete(self.y_train, self.labeled_index, axis=0)

        scores = []

        meta_examples = []

        for idx in range(n_queries):

            u_pool_size = np.size(u_y_pool)

            if u_pool_size <= 0:
                break

            # Extração de metafeatures dos dados não rotulados

            # Extração de medidas não supervisionadas
            uns_mfs = self._extract_unsupervised_mfs(u_X_pool)

            clst_mfs = self._extract_clustering_mfs(u_X_pool)

            mfs = pd.concat([uns_mfs, clst_mfs])

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                query_index, score, strategy_name = self._topline_query(
                    estimator=estimator,
                    query_strategies=query_strategies,
                    l_pool=(l_X_pool, l_y_pool),
                    u_pool=(u_X_pool, u_y_pool),
                    batch_size=5,
                    committee_size=committee_size)

            mfs['dataset_id'] = int(self.dataset_id)
            mfs['query_number'] = idx
            mfs['estimator'] = type(estimator).__name__
            mfs['best_strategy'] = strategy_name
            mfs['best_score'] = score

            meta_examples.append(mfs)

            new_X, new_y = u_X_pool[query_index], u_y_pool[query_index]

            l_X_pool = np.append(l_X_pool, new_X, axis=0)
            l_y_pool = np.append(l_y_pool, new_y, axis=0)

            u_X_pool = np.delete(u_X_pool, query_index, axis=0)
            u_y_pool = np.delete(u_y_pool, query_index, axis=0)

            scores.append(score)

        return pd.DataFrame(meta_examples).set_index('query_number')





if __name__ == "__main__":

    import os
    import logging
    from multiprocessing import Pool

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    from modAL import uncertainty as u
    from modAL import disagreement as d
    from modAL import batch as b

    DOWNLOAD_PATH = 'metabase/'

    class SVCLinear(SVC):
        pass

    def gen_metabase(args):
        try:
            logging.warning(f'[{args}] Iniciando construção de metabase.')
            dataset_id, estimator = args
            builder = MetaBaseBuilder(dataset_id=dataset_id, **init_args)

            df = builder.run(estimator=estimator, **run_args)

            try:
                os.mkdir(os.path.join(DOWNLOAD_PATH, str(dataset_id)))
            except FileExistsError:
                pass

            df.to_csv(os.path.join(
                DOWNLOAD_PATH,
                str(dataset_id),
                f'{type(estimator).__name__}.csv' ))

            logging.warning(f'[{args}] Metabase construida.')

        finally:
            pass
        # except Exception as e:
        #     logging.error(f'[{args}] Ocorreu um erro: {e}')


    query_strategies = [u.uncertainty_sampling,
                        u.margin_sampling,
                        u.entropy_sampling,
                        b.uncertainty_batch_sampling,
                        d.max_disagreement_sampling,
                        d.consensus_entropy_sampling,
                        d.vote_entropy_sampling]


    dataset_ids = {int(f.split('_')[0])
                   for f in os.listdir('../metabase/')
                   if f.endswith('.csv')}


    dataset_ids.update(int(line) for line in
                       open('selected_dataset_ids.txt'))

    clf_list = [SVCLinear(kernel='linear', probability=True),
                SVC(probability=True),
                RandomForestClassifier(),
                KNeighborsClassifier(),
                MLPClassifier(),
                LogisticRegression(),
                DecisionTreeClassifier(),
                GaussianNB()]


    init_args = {"random_state": 42, "l_size": 5}
    run_args = {"query_strategies": query_strategies,
                "n_queries": 100,
                "batch_size": 5}

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        # filename='active_learning.log'
                        )

    # result = list(map(gen_metabase, product(dataset_ids, clf_list)))

    with Pool() as p:
        result = p.map(gen_metabase, product(dataset_ids, clf_list))
