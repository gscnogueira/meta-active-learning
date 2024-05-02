from functools import partial
from itertools import product
from multiprocessing import Pool
import logging
import os


from meta_base_builder import MetaBaseBuilder

DOWNLOAD_PATH = 'metabase/'





def gen_metabase(dataset_id,
                 estimator,
                 query_strategies,
                 random_state,
                 initial_labeled_size,
                 n_queries,
                 batch_size):

    pid = os.getpid()

    context_string = f'{dataset_id}, {estimator}'

    try:
        logging.warning(f'[{context_string}, {pid}] Iniciando construção de metabase.')

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

        builder.run(estimator=estimator,
                    download_path=DOWNLOAD_PATH,
                    query_strategies=query_strategies)

        logging.warning(f'[{context_string}] Metabase construida.')

    except Exception as e:
        logging.error(f'[{context_string}] Ocorreu um erro: {e}')



if __name__ == '__main__':

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from modAL.uncertainty import margin_sampling
    from modAL.disagreement import consensus_entropy_sampling

    from expected_error import expected_error_reduction
    from information_density import (density_weighted_sampling,
                                     training_utility_sampling)

    class SVCLinear(SVC):
        pass

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    dataset_ids = {40, 41}

    clf_list = [
        SVCLinear(kernel='linear', probability=True),
        SVC(probability=True),
        RandomForestClassifier(),
        KNeighborsClassifier(),
        MLPClassifier(),
        LogisticRegression(),
        DecisionTreeClassifier(),
        GaussianNB()
    ]

    query_strategies = [
        training_utility_sampling,
        density_weighted_sampling,
        margin_sampling,
        consensus_entropy_sampling,
        expected_error_reduction
    ]

    gen_metabase_partial = partial(
        gen_metabase,
        query_strategies=query_strategies,
        initial_labeled_size=5,
        n_queries=10,
        batch_size=1,
        random_state=42)

    list(map(lambda args: gen_metabase_partial(*args),
             product(dataset_ids, clf_list)))
