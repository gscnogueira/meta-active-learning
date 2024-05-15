from os import environ
# environ['OMP_NUM_THREADS'] = '1'

import os
from functools import partial

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
from tqdm import tqdm

from active_learning import ActiveLearningExperiment
import config


def random_sampling(classifier, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)),
                                 size=n_instances,
                                 replace=False)
    return query_idx, X[query_idx]


def gen_meta_base(data_path, estimator):

    dfs = []
    index_columns = ['dataset_id', 'query_number']

    for root, _, files in os.walk(data_path):
        file_path = f'{estimator.__name__}.csv'

        if file_path not in files:
            continue

        df = pd.read_csv(os.path.join(root, file_path))

        # remove coluna sem nome gerada por bug no metabasebuilder
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # remove coluna que não é utilizada no treinemanto (por enquanto)
        df.drop(columns=['estimator'], inplace=True)

        df.query_number = df.query_number.astype(int)
        df.dataset_id = df.dataset_id.astype(int)

        dfs.append(df)

    dfs_filtered = [df for df in dfs if df.query_number.max() == 99
                    and len(df) == 100]

    print(f'Foram identificadas {len(dfs_filtered)} bases uteis')

    final_df = pd.concat(dfs_filtered, join='inner')
    final_df.set_index(index_columns, inplace=True)

    return final_df


def preprocess_meta_base(meta_base: pd.DataFrame) -> pd.DataFrame:
    # substitui valores infinitos com nan
    processed_meta_base = meta_base.replace([np.inf, -np.inf], np.nan)

    # Remove mfts que apresentas valor NaN em todos os conjuntos
    processed_meta_base.drop(columns=['num_to_cat'], inplace=True)

    return processed_meta_base


def split_train_data(meta_base, train_index):

    to_drop_on_training = ['dataset_id', 'best_strategy', 'best_score']
    train_data = meta_base.loc[train_index].reset_index()

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


def run_experiment(estimator, X_train, y_train, test_data_id,
                   initial_labeled_size, n_queries, batch_size, random_state,
                   **kwargs):

    context_string = f'({test_data_id}, {os.getpid()})'
    print(context_string, 'Experimento iniciado.', flush=True)

    exp = ActiveLearningExperiment(dataset_id=test_data_id,
                                   initial_labeled_size=N_LABELED_START,
                                   n_queries=N_QUERIES,
                                   batch_size=BATCH_SIZE,
                                   random_state=RANDOM_STATE)

    metrics_dict = dict()

    # ESTRATÉGIAS CLASSICAS
    for strategy in [random_sampling] + config.query_strategies:
        print(context_string, f'Rodando {strategy.__name__}', flush=True)
        scores = exp.run(estimator=estimator(**kwargs), query_strategy=strategy)
        metrics_dict[strategy.__name__] = scores

    # RANDOM META-SAMPLING
    print(context_string, 'Rodando random_meta_sampling', flush=True)
    random_results = exp.run_baseline(estimator=estimator(**kwargs),
                                      query_strategies=config.query_strategies)

    metrics_dict['random_meta_sampling'] = random_results.scores
    metrics_dict['random_sampling_choice'] = random_results.choices

    # PERFECT META-SAMPLING
    print(context_string, 'Rodando perfect_meta_sampling', flush=True)
    perfect_results = exp.run_topline(estimator=estimator(**kwargs),
                                      query_strategies=config.query_strategies)

    metrics_dict['perfect_meta_sampling'] = perfect_results.scores
    metrics_dict['perfect_sampling_choice'] = perfect_results.choices

    # META-SAMPLING
    print(context_string, 'Rodando meta_sampling', flush=True)
    ms_results = exp.run_meta_query(estimator=estimator(**kwargs),
                                    meta_X=X_train, meta_y=y_train)

    metrics_dict['meta_sampling'] = ms_results.scores
    meta_sampling_real_choices = [c.pred for c in ms_results.choices]
    meta_sampling_ideal_choices = [c.true for c in ms_results.choices]

    metrics_dict['meta_sampling_choice'] = meta_sampling_real_choices
    metrics_dict['meta_sampling_ideal_choice'] = meta_sampling_ideal_choices

    print(context_string, 'Experimento finalizado.', flush=True)

    return metrics_dict


def run_split(meta_base, split):

    train_data_ids, test_data_id = split

    download_path = os.path.join('results', ESTIMATOR.__name__)
    csv_file = os.path.join(download_path, f'{test_data_id}.csv')

    try:
        os.mkdir(download_path)
    except FileExistsError:
        pass

    X_train, y_train = split_train_data(meta_base, train_data_ids)

    metrics = run_experiment(
        X_train=X_train,
        y_train=y_train,
        test_data_id=test_data_id,
        estimator=ESTIMATOR,
        initial_labeled_size=N_LABELED_START,
        n_queries=N_QUERIES,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE)

    df = pd.DataFrame(metrics)
    df.to_csv(csv_file)

if __name__ == '__main__':

    import sys
    from multiprocessing import Pool
    from functools import partial

    DATA_DIR = 'metabase/'
    BATCH_SIZE = 1
    N_LABELED_START = 5
    RANDOM_STATE = 42
    N_QUERIES = 100 

    ESTIMATOR = GaussianNB

    meta_base = gen_meta_base(DATA_DIR, ESTIMATOR)

    meta_base = preprocess_meta_base(meta_base)

    loo = LeaveOneOut()
    dataset_ids = meta_base.index.levels[0]

    splits = [(dataset_ids[train_index], int(dataset_ids[test_index][0]))
              for train_index, test_index in loo.split(dataset_ids)]

    n_workers = 48  

    run_split_partial = partial(run_split, meta_base)

    with Pool(n_workers) as p:
        results = [e for e in tqdm(p.imap_unordered(run_split_partial, splits),
            total=len(splits),
            file=sys.stdout)]
