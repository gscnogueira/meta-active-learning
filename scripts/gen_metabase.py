from os import environ
# environ['OMP_NUM_THREADS'] = '1'

import time
import sys
from functools import partial
from multiprocessing import Pool, get_context
import logging
import os

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from modAL.uncertainty import margin_sampling

from meta_base_builder import MetaBaseBuilder
from expected_error import expected_error_reduction
from information_density import (density_weighted_sampling,
                                 training_utility_sampling)

DOWNLOAD_PATH = '../metabase/'

estimator_dict = {
    "KNN": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
}

ERROR_LIST = []

def gen_metabase(dataset_id,
                 estimator_name, 
                 query_strategies,
                 random_state,
                 initial_labeled_size,
                 n_queries,
                 batch_size):

    estimator = estimator_dict[estimator_name]()

    pid = os.getpid()

    context_string = f'{dataset_id}, {estimator}, {pid}'

    try:
        logging.warning(f'[{context_string}] Iniciando construção de metabase.')

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

        logging.warning(f'[{context_string}] Metabase construida.')

    except Exception as e:
        ERROR_LIST.append(dataset_id)
        logging.error(f'[{context_string}] Ocorreu um erro: {e}')



if __name__ == '__main__':

    DATASETS_PATH = '../datasets'
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    clf_list = ["KNN", "GaussianNB"]

    query_strategies = [
        training_utility_sampling,
        density_weighted_sampling,
        margin_sampling,
        expected_error_reduction]

    gen_metabase_partial = partial(
        gen_metabase,
        query_strategies=query_strategies,
        initial_labeled_size=5,
        n_queries=2,  # NAO ESQUECE DE MUDAR PRA 100 DE NOVO!!!!!!
        batch_size=1,
        random_state=42)

    dataset_ids = list(os.path.join(DATASETS_PATH, f)
                       for f in os.listdir(DATASETS_PATH))

    n_workers = 1  # MUDAR DEPOIS

    for estimator_name in clf_list:
        t = time.time()
        print(f'Iniciando Criação de metabase para {estimator_name}')
        for d in tqdm(dataset_ids):
            gen_metabase_partial(d, estimator_name)
        t = time.time() - t
        print(f'{estimator_name}: Done in {t} seconds!')
        break

    print(ERROR_LIST)

    exit()

    with get_context("fork").Pool(n_workers) as p:
        results = [e for e in tqdm(p.imap_unordered(gen_metabase_partial, args),
                                   total=len(args),
                                   file=sys.stdout)]
