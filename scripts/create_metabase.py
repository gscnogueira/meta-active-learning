import os
import warnings
from multiprocessing import Pool

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import openml
import pandas as pd

from metabase_builder import MetaBaseBuilder
from config import dataset_ids


class SVCLinear(SVC):
    pass



clf_list = [SVCLinear(kernel='linear', probability=True),
            SVC(probability=True),
            RandomForestClassifier(),
            KNeighborsClassifier(),
            MLPClassifier(),
            LogisticRegression(),
            DecisionTreeClassifier(),
            GaussianNB(),
            ]


query_strategies = (MetaBaseBuilder.uncertainty_strategies
                    + MetaBaseBuilder.disagreement_strategies)


def download_metabase(dataset):

    builder = MetaBaseBuilder(estimators=clf_list,
                              query_strategies=query_strategies,
                              n_queries=100,
                              initial_l_size=5,
                              batch_size=5)

    metabase = builder.build(dataset)

    file_name = f'{dataset.id}_{dataset.name}.csv'
    data_path = '../metabase'
    
    metabase.to_csv(os.path.join(data_path, file_name))


if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        datasets = openml.datasets.get_datasets(dataset_ids)


    with Pool() as p:
        results = p.map(download_metabase, datasets)