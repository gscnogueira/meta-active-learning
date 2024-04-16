from itertools import product
from multiprocessing import Pool
import logging
import os

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

from active_learning import MetaBaseBuilder

DOWNLOAD_PATH = 'metabase/'

query_strategies = [u.uncertainty_sampling,
                    u.margin_sampling,
                    u.entropy_sampling,
                    b.uncertainty_batch_sampling,
                    d.max_disagreement_sampling,
                    d.consensus_entropy_sampling,
                    d.vote_entropy_sampling]


class SVCLinear(SVC):
    pass


def gen_metabase(args):
    init_args = {"random_state": 42,
                 "initial_labeled_size": 5,
                 "n_queries": 5,
                 "batch_size": 5}

    run_args = {"query_strategies": query_strategies}

    pid = os.getpid()

    try:
        logging.warning(f'[{args}, {pid}] Iniciando construção de metabase.')
        dataset_id, estimator = args
        builder = MetaBaseBuilder(dataset_id=dataset_id, **init_args)

        try:
            os.mkdir(os.path.join(DOWNLOAD_PATH, str(dataset_id)))
        except FileExistsError:
            pass

        builder.run(estimator=estimator,
                    download_path=DOWNLOAD_PATH,
                    **run_args)

        logging.warning(f'[{args}, {pid}] Metabase construida.')

    except Exception as e:
        raise e
        logging.error(f'[{args}, {pid}] Ocorreu um erro: {e}')


clf_list = [SVCLinear(kernel='linear', probability=True),
            SVC(probability=True),
            RandomForestClassifier(),
            KNeighborsClassifier(),
            MLPClassifier(),
            LogisticRegression(),
            DecisionTreeClassifier(),
            GaussianNB()]

dataset_ids = {int(f.split('_')[0])
               for f in os.listdir('../metabase/')
               if f.endswith('.csv')}

dataset_ids.update(int(line) for line in
                   open('selected_dataset_ids.txt'))

finished_datasets = {
    int(id) for id in os.listdir('metabase/')
    if len(os.listdir(os.path.join('metabase', id))) == len(clf_list)}

dataset_ids -= finished_datasets

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s:%(levelname)s:%(message)s')

with Pool() as p:
    result = p.map(gen_metabase, product(dataset_ids, clf_list))
