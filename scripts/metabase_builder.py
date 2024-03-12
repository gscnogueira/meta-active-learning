from functools import partial
from typing import Union
import warnings
import logging

from modAL.models import ActiveLearner, Committee
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pymfe.mfe import MFE
import numpy as np
import openml
import pandas as pd

# Query Strategies
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from modAL.disagreement import vote_entropy_sampling, consensus_entropy_sampling, max_disagreement_sampling
from modAL.batch import uncertainty_batch_sampling


class MetaBaseBuilder:
    uncertainty_strategies = [
        uncertainty_sampling,
        uncertainty_batch_sampling,
        margin_sampling,
        entropy_sampling,
    ]
    disagreement_strategies = [
        vote_entropy_sampling,
        consensus_entropy_sampling,
        max_disagreement_sampling
    ]

    def __init__(self, estimators,
                 query_strategies,
                 n_queries=50,
                 batch_size=5,
                 committee_size=3,
                 initial_l_size=-1, ):
        self.__estimators = estimators
        self.__query_strategies = query_strategies
        self.__n_queries = n_queries
        self.__batch_size = batch_size
        self.__committee_size = committee_size
        self.__initial_l_size = initial_l_size

        self.__metabase = list()

        self.logger = logging.getLogger('MetaBaseBuilder')
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)

    
    @property
    def metabase(self):
        return pd.DataFrame.from_records(self.__metabase)

    def fit(self, dataset: openml.datasets.OpenMLDataset):
        pass

    def build(self):
        # Faz tratamento de dados
        X, y = self.__load_data(dataset)

        self.classes_ = np.unique(y)
        self.dataset = dataset

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

        # Faz com que inicialmente haja uma instância rotulada por classe
        labeled_index = [np.random.choice(np.where(y_train == cls)[0])
                         for cls in np.unique(y_train)]

        if len(labeled_index) < self.__initial_l_size:
            additional_index = np.random.choice(len(y_train),
                                                size=self.__initial_l_size,
                                                replace=False)
            labeled_index.extend(additional_index.tolist())

        l_X_pool = X_train[labeled_index]
        l_y_pool = y_train[labeled_index]

        u_X_pool = np.delete(X_train, labeled_index, axis=0)
        u_y_pool = np.delete(y_train, labeled_index, axis=0)

        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            learners = self.__build_learners(l_X_pool, l_y_pool)

            for learner in learners:
                self.__teach_learner(learner, u_X_pool, u_y_pool,
                                     X_test, y_test)

        self.logger.info(f"{dataset.id}_{dataset.name} - Metabase para  criada.")
        return self.metabase

    def __load_data(self, dataset: openml.datasets.OpenMLDataset):
        X, y, categorical_indicator, _ = dataset.get_data(
            target=dataset.default_target_attribute)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers = [('one-hot-encoder', encoder, categorical_indicator)]

        preprocessor = ColumnTransformer(transformers, remainder='passthrough')

        le = LabelEncoder()

        X, y = preprocessor.fit_transform(X), le.fit_transform(y)

        return X, y

    def __build_learners(self, X_training, y_training):
        return [self.__build_learner(estimator, strategy,
                                     X_training, y_training)
                for estimator in self.__estimators
                for strategy in self.__query_strategies]

    def __build_learner(self, estimator, strategy,
                        X_training, y_training):

        query_strategy = partial(strategy,
                                 n_instances=self.__batch_size)

        # Copia o nome para facilitar debbug
        query_strategy.__name__ = strategy.__name__

        args = {
            'estimator': estimator,
            'query_strategy': query_strategy,
            'X_training': X_training,
            'y_training': y_training
        }

        if strategy in self.uncertainty_strategies:
            return ActiveLearner(**args)
        else:
            learner_list = list()

            args.pop('query_strategy')
            for _ in range(self.__committee_size):
                learner = ActiveLearner(**args)
                learner_list.append(learner)

            committee = Committee(learner_list=learner_list,
                                  query_strategy=query_strategy)
            return committee

    def __teach_learner(self, learner: Union[ActiveLearner, Committee],
                        X_pool, y_pool, X_test, y_test):


        context_string = (f"{self.dataset.id}_{self.dataset.name}::"
                          f"{self.__get_estimator_name(learner)}::"
                          f"{learner.query_strategy.__name__} -")

        for idx in range(self.__n_queries):

            query_index, query_instance = learner.query(X_pool)

            self.logger.info(f"{context_string} Criando instância para query {idx}...")

            learner.teach(X=X_pool[query_index], y=y_pool[query_index])


            metainstance = self.__create_metainstance(
                learner=learner,
                X_pool=X_pool, y_pool=y_pool,
                X_test=X_test, y_test=y_test)

            self.__metabase.append(metainstance)
            
            X_pool = np.delete(X_pool, query_index, axis=0)
            y_pool = np.delete(y_pool, query_index, axis=0)

            self.logger.info(f"{context_string} Instância criada "
                             f"[L:{self.__get_labeled_pool_size(learner)},U:{y_pool.shape[0]}]")

            if np.size(y_pool) == 0:
                break

    def __create_metainstance(self, learner, X_pool, y_pool, X_test, y_test):
        scores = self.__eval_learner(learner, X_test, y_test)
        metafeatures = self.__extract_metafeatures(X_pool, y_pool)
        metainstance = dict()

        metainstance.update(scores)
        metainstance.update(metafeatures)

        return metainstance

    def __extract_metafeatures(self, X, y):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mfe = MFE(groups='all')
            mfe.fit(X, y)
            ft = mfe.extract()

        return {k: v for k, v in zip(*ft)}

    def __eval_learner(self, learner: ActiveLearner, X, y_true):
        y_pred = learner.predict(X)
        ys = (y_pred, y_true)
        scores = dict()

        scores['estimator'] = self.__get_estimator_name(learner)
        scores['query-strategy'] = learner.query_strategy.__name__
        scores['accuracy'] = metrics.accuracy_score(*ys)
        scores['f1-micro'] = metrics.f1_score(*ys, average='micro')
        scores['f1-macro'] = metrics.f1_score(*ys, average='macro')
        scores['f1-weighted'] = metrics.f1_score(*ys, average='weighted')

        return scores

    def __get_estimator_name(self, learner: Union[ActiveLearner, Committee]):
        if isinstance(learner, ActiveLearner):
            return type(learner.estimator).__name__
        if isinstance(learner, Committee):
            return type(learner.learner_list[0].estimator).__name__

    def __get_labeled_pool_size(self, learner: Union[ActiveLearner, Committee]):
        if isinstance(learner, ActiveLearner):
            return learner.X_training.shape[0]
        if isinstance(learner, Committee):
            return learner.learner_list[0].X_training.shape[0]


if __name__  == '__main__':

    from sklearn.svm import SVC


    clf_list = [SVC(probability=True)]

    query_strategies = [uncertainty_sampling]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        dataset = openml.datasets.get_dataset(40)

    builder = MetaBaseBuilder(estimators=clf_list,
                              query_strategies=query_strategies,
                              n_queries=1,
                              initial_l_size=5,
                              batch_size=5)

    print(builder.build(dataset))
