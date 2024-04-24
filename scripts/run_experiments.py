import os
import warnings
import pickle as pkl

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
from modAL import uncertainty as u, disagreement as d, batch as b
from tqdm import tqdm

from active_learning import ActiveLearningExperiment

def gen_meta_base(data_path, estimator):

    dfs = []
    index_columns = ['dataset_id','estimator', 'query_number']

    for root, _, files in os.walk(DATA_DIR):
        
        file_path = f'{estimator.__name__}.csv'

        if  file_path not in files:
            continue
        
        df = pd.read_csv(os.path.join(root, file_path))

        df.query_number = df.query_number.astype(int)
        df.dataset_id = df.dataset_id.astype(int)

        df.set_index(index_columns, inplace=True)
        
        dfs.append(df)
        
    return pd.concat(dfs, join='inner')


def preprocess_meta_base(meta_base: pd.DataFrame) -> pd.DataFrame:

    # substitui valores infinitos com nan
    processed_meta_base = meta_base.replace([np.inf, -np.inf], np.nan)

    return processed_meta_base

def split_train_data(meta_base, train_index):

    train_data = meta_base.loc[train_index].reset_index()

    to_drop_on_training = ['dataset_id', 'best_strategy', 'best_score', 'estimator', 'query_number']

    X_train = train_data.drop(columns=to_drop_on_training)
    y_train = train_data['best_strategy']

    return X_train, y_train


def gen_meta_model(X_train, y_train):

    meta_model = Pipeline([
        ('mean-inputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('meta-model', RandomForestClassifier())
    ])

    meta_model.fit(X_train.values, y_train)

    return meta_model


def run_experiment(estimator, train_data, test_data, initial_labeled_size,
                   n_queries, batch_size, random_state, **kwargs):


    X_train, y_train = split_train_data(meta_base, train_data)
    test_data_id = int(test_data[0])

    print('Conjunto de Teste:', test_data_id)

    meta_model = gen_meta_model(X_train, y_train)

    exp = ActiveLearningExperiment(dataset_id=test_data_id,
                                   initial_labeled_size=N_LABELED_START,
                                   n_queries=N_QUERIES,
                                   batch_size=BATCH_SIZE,
                                   random_state=RANDOM_STATE)

    print('Conjunto de treino:', exp.X_train.shape, f'[|L| = {len(exp.labeled_index)}]')
    print('Conjunto de teste:', exp.X_test.shape)


    query_strategies = [u.uncertainty_sampling,
                        u.entropy_sampling,
                        u.margin_sampling,
                        b.uncertainty_batch_sampling,
                        d.consensus_entropy_sampling,
                        d.vote_entropy_sampling,
                        d.max_disagreement_sampling]


    metrics_dict = dict()

    for strategy in query_strategies:
        scores = exp.run(estimator=estimator(**kwargs), query_strategy=strategy)
        metrics_dict[strategy.__name__] = scores

    metrics_dict['random_meta_sampling'], metrics_dict['random_sampling_choice'] = exp.run_baseline(
        estimator=estimator(**kwargs), query_strategies=query_strategies)

    metrics_dict['perfect_meta_sampling'], metrics_dict['perfect_sampling_choice'] = exp.run_topline(
        estimator=estimator(**kwargs), query_strategies=query_strategies)

    metrics_dict['meta_sampling'], metrics_dict['meta_sampling_choice'] = exp.run_meta_query(
        estimator=estimator(**kwargs), meta_model=meta_model)

    return pd.DataFrame(metrics_dict)


if __name__ == '__main__':

    DATA_DIR = 'metabase/'
    BATCH_SIZE = 5
    N_LABELED_START = 5
    RANDOM_STATE = 42
    N_QUERIES = 100

    ESTIMATOR = KNeighborsClassifier

    meta_base = gen_meta_base(DATA_DIR, ESTIMATOR) 

    # Remove mft que apresenta valor NaN em todos os conjuntos
    meta_base.drop(columns = ['num_to_cat'], inplace=True)

    meta_base = preprocess_meta_base(meta_base)

    dataset_ids = meta_base.index.levels[0]

    loo = LeaveOneOut()

    print(f'Iniciando processo para {len(dataset_ids)} bases.')

    for i, (train_index, test_index) in enumerate(loo.split(dataset_ids)):
        print(f'Split [{i}/{len(dataset_ids)}]')

        train_data = dataset_ids[train_index]
        test_data = dataset_ids[test_index]

        download_path = os.path.join('results', ESTIMATOR.__name__)
        csv_file = os.path.join(download_path, f'{test_data[0]}.csv')

        try:
            df = run_experiment(
                train_data=train_data,
                test_data=test_data,
                estimator=ESTIMATOR,
                initial_labeled_size=N_LABELED_START,
                n_queries=N_QUERIES,
                batch_size=BATCH_SIZE,
                random_state=RANDOM_STATE)
        except Exception as e:
            print('Ocorreu um erro:', e)
            continue

        try:
            os.mkdir(download_path)
        except FileExistsError:
            pass

        df.to_csv(csv_file)
