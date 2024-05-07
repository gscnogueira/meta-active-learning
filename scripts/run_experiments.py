import os
import warnings
import pickle as pkl

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
from modAL import uncertainty as u, disagreement as d, batch as b
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

    metrics_dict = dict()

    for strategy in [random_sampling] + config.query_strategies:
        scores = exp.run(estimator=estimator(**kwargs), query_strategy=strategy)
        metrics_dict[strategy.__name__] = scores

    metrics_dict['random_meta_sampling'], metrics_dict['random_sampling_choice'] = exp.run_baseline(
        estimator=estimator(**kwargs), query_strategies=config.query_strategies)

    metrics_dict['perfect_meta_sampling'], metrics_dict['perfect_sampling_choice'] = exp.run_topline(
        estimator=estimator(**kwargs), query_strategies=config.query_strategies)

    metrics_dict['meta_sampling'], metrics_dict['meta_sampling_choice'] = exp.run_meta_query(
        estimator=estimator(**kwargs), meta_model=meta_model)

    return metrics_dict


def run_split(train_data, test_data):
    download_path = os.path.join('results', ESTIMATOR.__name__)
    csv_file = os.path.join(download_path, f'{test_data[0]}.csv')

    try:
        os.mkdir(download_path)
    except FileExistsError:
        pass

    metrics = run_experiment(
        train_data=train_data,
        test_data=test_data,
        estimator=ESTIMATOR,
        initial_labeled_size=N_LABELED_START,
        n_queries=N_QUERIES,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE)

    df = pd.DataFrame(metrics)
    df.to_csv(csv_file)


if __name__ == '__main__':

    DATA_DIR = 'metabase/'
    BATCH_SIZE = 1
    N_LABELED_START = 5
    RANDOM_STATE = 42
    N_QUERIES = 100

    ESTIMATOR = GaussianNB

    meta_base = gen_meta_base(DATA_DIR, ESTIMATOR)

    # Remove mfts que apresentas valor NaN em todos os conjuntos
    meta_base.drop(columns=['num_to_cat'], inplace=True)

    meta_base = preprocess_meta_base(meta_base)

    dataset_ids = meta_base.index.levels[0]

    loo = LeaveOneOut()

    train_index, test_index = next(loo.split(dataset_ids))

    train_data = dataset_ids[train_index]
    test_data = dataset_ids[test_index]

    run_split(train_data, test_data)
