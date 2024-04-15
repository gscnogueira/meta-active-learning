import os
import warnings
import pickle as pkl

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
import matplotlib.pyplot
from modAL import uncertainty as u, disagreement as d, batch as b
from tqdm import tqdm

from active_learning import ActiveLearningExperiment

def gen_meta_base(data_path):

    dfs = []
    index_columns = ['dataset_id','estimator', 'query_number']

    for root, _, files in os.walk(DATA_DIR):
        
        if not len(files):
            continue
        
        df = pd.concat(objs=(pd.read_csv(os.path.join(root,file)) for file in files))

        df.query_number = df.query_number.astype(int)
        df.dataset_id = df.dataset_id.astype(int)

        df.set_index(index_columns, inplace=True)
        
        dfs.append(df)
        
    return pd.concat(dfs)


def preprocess_meta_base(meta_base: pd.DataFrame) -> pd.DataFrame:

    # substitui valores infinitos com nan
    processed_meta_base = meta_base.replace([np.inf, -np.inf], np.nan)

    return processed_meta_base

def split_train_data(meta_base, train_index):

    train_data = meta_base.loc[train_index].xs("SVC", level='estimator').reset_index()

    to_drop_on_training = ['dataset_id', 'best_strategy', 'best_score']

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


def run_experiment(train_data, test_data, initial_labeled_size,
                   n_queries, batch_size, random_state):


    X_train, y_train = split_train_data(meta_base, train_data)
    test_data_id = int(test_data[0])

    print(X_train.shape, y_train.shape)
    print(test_data_id)

    meta_model = gen_meta_model(X_train, y_train)

    with open(os.path.join('meta_models', f'{test_data_id}.pkl'), 'wb') as f:
        pkl.dump(meta_model, f)


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

    # for strategy in query_strategies:
    #     scores = exp.run(estimator=SVC(probability=True), query_strategy=strategy)
    #     metrics_dict[strategy.__name__] = scores

    metrics_dict['random_meta_sampling'], metrics_dict['random_sampling_choice'] = exp.run_baseline(
        estimator=SVC(probability=True), query_strategies=query_strategies)

    metrics_dict['perfect_meta_sampling'], metrics_dict['perfect_sampling_choice'] = exp.run_topline(
        estimator=SVC(probability=True), query_strategies=query_strategies)

    metrics_dict['meta_sampling'], metrics_dict['meta_sampling_choice'] = exp.run_meta_query(
        estimator=SVC(probability=True), meta_model=meta_model)

    return pd.DataFrame(metrics_dict)


if __name__ == '__main__':

    DATA_DIR = 'metabase/'
    BATCH_SIZE = 5
    N_LABELED_START = 5
    RANDOM_STATE = 42
    N_QUERIES = 2

    meta_base = gen_meta_base(DATA_DIR)

    # Remove mft que apresenta valor NaN em todos os conjuntos
    meta_base.drop(columns = ['num_to_cat'], inplace=True)

    meta_base = preprocess_meta_base(meta_base)

    dataset_ids = meta_base.index.levels[0]

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(dataset_ids):
        train_data = dataset_ids[train_index]
        test_data = dataset_ids[test_index]

        df = run_experiment(train_data=train_data,
                            test_data=test_data,
                            initial_labeled_size=N_LABELED_START,
                            n_queries=N_QUERIES,
                            batch_size=BATCH_SIZE,
                            random_state=RANDOM_STATE)
        df.to_csv(os.path.join('results', f'{test_data[0]}.csv'))


