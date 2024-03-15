import random
import openml
import warnings 
from multiprocessing import Pool

from tqdm import tqdm 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold



already_created = [
    991,
    917,
    799,
    740,
    44620,
    44616,
    44493,
    44433,
    44422,
    44397,
    44383,
    44382,
    44370,
    40704,
    1016,
]


def get_dataset_ids(max_number_of_instances=10000,
                    min_number_of_instances=100,
                    max_number_of_features=500,
                    max_number_of_instances_with_missing_values=0,
                    max_number_of_classes=5,
                    min_number_of_classes=2):

    df = openml.datasets.list_datasets(output_format='dataframe')

    selected_datasets = df[
        (df.status == 'active') &
        (df.NumberOfFeatures <= max_number_of_features) &
        (df.NumberOfInstancesWithMissingValues
         == max_number_of_instances_with_missing_values) &
        (df.NumberOfInstances.between(min_number_of_instances,
                                      max_number_of_instances)) &
        (df.NumberOfClasses.between(min_number_of_classes,
                                    max_number_of_classes))
    ]

    return selected_datasets.did.tolist()

def test_dataset(dataset_id):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dataset = openml.datasets.get_dataset(dataset_id)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            X, y, categorical_indicator, _ = dataset.get_data(
                target=dataset.default_target_attribute)

            categorical_encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')
            
            transformer = ColumnTransformer([
                ('onehot_encoder', categorical_encoder, categorical_indicator)],
                                            remainder='passthrough'
                                            )
            
            model = make_pipeline(transformer, RandomForestClassifier())
            
            skf = StratifiedKFold(n_splits=5)

            cross_validate(model, X, y,
                           cv=skf,
                           error_score='raise')
    except Exception as e:
        return False

    return True

if __name__ == '__main__':

    N_DATASETS = 100 - len(already_created)


    dataset_ids = get_dataset_ids()

    with Pool() as pool:
        results = pool.map(test_dataset, dataset_ids)

    filtered_ids = [id for id, result in zip(dataset_ids, results)
                    if result and id not in already_created]

    random.seed(42)

    selected_ids = sorted(random.sample(filtered_ids, N_DATASETS))

    with open('selected_dataset_ids.txt', 'w') as arquivo:
        for id in selected_ids:
            arquivo.write(str(id) + '\n')

