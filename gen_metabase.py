"""Script para a geração de metabase."""

from functools import partial
from multiprocessing import get_context
from os import environ
import logging
import os
import re
import sys

environ['OMP_NUM_THREADS'] = '1'

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from modAL.uncertainty import margin_sampling

from utils.meta_base_builder import MetaBaseBuilder
import config


estimator_dict = {
    "KNN": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
}


def gen_metabase(dataset_id, estimator_name,
                 query_strategies,
                 random_state,
                 initial_labeled_size,
                 n_queries,
                 batch_size):

    estimator = estimator_dict[estimator_name]()

    pid = os.getpid()

    context_string = f'{os.path.basename(dataset_id)}, {estimator}, {pid}'

    try:
        # logging.warning(f'[{context_string}] Iniciando construção de metabase.')

        builder = MetaBaseBuilder(dataset_id=dataset_id,
                                  initial_labeled_size=initial_labeled_size,
                                  n_queries=n_queries,
                                  batch_size=batch_size,
                                  random_state=random_state)

        dir_path = os.path.join(DOWNLOAD_PATH, builder.dataset_id)

        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass

        builder.run(estimator=estimator,
                    download_path=DOWNLOAD_PATH,
                    query_strategies=query_strategies)

        # logging.warning(f'[{context_string}] Metabase construida.')

    except Exception as e:
        logging.error(f'[{context_string}] Ocorreu um erro: {e}')


if __name__ == '__main__':

    DATASETS_PATH = 'datasets/'
    DOWNLOAD_PATH = 'metabase/'
    N_WORKERS = 1

    kwargs = {
        'query_strategies': config.query_strategies,
        'initial_labeled_size': 5,
        'n_queries': 100,
        'batch_size': 1,
        'random_state': 42
    }

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    dataset_ids = [os.path.join(DATASETS_PATH, f'{filename.strip()}.arff')
                   for filename in open('datasets.txt')]

    print(f'Foram encontrados {len(dataset_ids)} datasets.')

    for estimator_name in config.classifier_list:

        gen_metabase_partial = partial(gen_metabase,
                                       estimator_name=estimator_name, **kwargs)

        with get_context("fork").Pool(N_WORKERS) as p:
            generator = p.imap_unordered(gen_metabase_partial, dataset_ids)
            pbar = tqdm(generator, total=len(dataset_ids),
                        file=sys.stdout, desc=estimator_name)
            any(pbar)