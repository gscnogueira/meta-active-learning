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

from metabase_builder import MetaBaseBuilder


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
                              batch_size=5,
                              download_path="/dev/shm/metabase")


    builder.fit(dataset)
    builder.build()


if __name__ == '__main__':

    # dataset_ids = [int(line) for line in open('selected_dataset_ids.txt')]
    dataset_ids = [801]


    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        datasets = openml.datasets.get_datasets(dataset_ids)


    for dataset in datasets:
        download_metabase(dataset)

    # with Pool() as p:
    #     results = p.map(download_metabase, datasets)
