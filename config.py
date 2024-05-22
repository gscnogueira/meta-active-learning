from modAL.uncertainty import margin_sampling
from utils.expected_error import expected_error_reduction
from utils.information_density import (density_weighted_sampling,
                                       training_utility_sampling)

classifier_list = [
        # "KNN",
        "GaussianNB",
        "RandomForestClassifier",
        "SVC"
        ]

query_strategies = [
    training_utility_sampling,
    density_weighted_sampling,
    margin_sampling,
    expected_error_reduction
]

query_strategy_dict = {
    "training_utility_sampling": training_utility_sampling,
    "density_weighted_sampling": density_weighted_sampling,
    "margin_sampling": margin_sampling,
    "expected_error_reduction": expected_error_reduction
}
