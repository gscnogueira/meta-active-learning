from os import environ
environ['OMP_NUM_THREADS'] = '1'

from functools import partial
from itertools import product
from multiprocessing import Pool, get_context
import logging
import os

import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from modAL.uncertainty import margin_sampling

from meta_base_builder import MetaBaseBuilder
from expected_error import expected_error_reduction
from information_density import (density_weighted_sampling,
                                     training_utility_sampling)

DOWNLOAD_PATH = 'metabase/'

estimator_dict = {
    "KNN": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
    "DecisionTreeClassifier": DecisionTreeClassifier
}

def gen_metabase(args,
                 query_strategies,
                 random_state,
                 initial_labeled_size,
                 n_queries,
                 batch_size):

    dataset_id, estimator_name = args

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

        dir_path = os.path.join(DOWNLOAD_PATH, str(dataset_id)) 

        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass

        csv_file_name = os.path.join(dir_path, f'{type(estimator).__name__}.csv' )
        if os.path.exists(csv_file_name):
            if pd.read_csv(csv_file_name)['query_number'].max() >= 99 :
                logging.warning(f'[{context_string}] Metabase já havia sido gerada.')
                return

        builder.run(estimator=estimator,
                    download_path=DOWNLOAD_PATH,
                    query_strategies=query_strategies)

        logging.warning(f'[{context_string}] Metabase construida.')

    except Exception as e:
        logging.error(f'[{context_string}] Ocorreu um erro: {e}')



if __name__ == '__main__':


    class SVCLinear(SVC):
        pass

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    dataset_ids = [int(line) for line in open('selected_dataset_ids.txt')]

    clf_list = [
        "KNN",
        # "DecisionTreeClassifier" ,
        # "GaussianNB"
    ]

    query_strategies = [
        training_utility_sampling,
        density_weighted_sampling,
        margin_sampling,
        expected_error_reduction
    ]

    gen_metabase_partial = partial(
        gen_metabase,
        query_strategies=query_strategies,
        initial_labeled_size=5,
        n_queries=100, # NAO ESQUECE DE MUDAR PRA 100 DE NOVO!!!!!!
        batch_size=1,
        random_state=42)

    import time
    import sys

    t = time.time()

    args = list(product(dataset_ids, clf_list))
    n_workers = 48

    with get_context("fork").Pool(n_workers) as p:
        results = [e for e in tqdm(p.imap_unordered(gen_metabase_partial, args),
                                   total=len(args),
                                   file=sys.stdout)]

    t = time.time() - t

    print(f'Done in {t} seconds!')
