import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from active_learning import ActiveLearningExperiment


class MetaBaseBuilder(ActiveLearningExperiment):

    def run(self, estimator,
            query_strategies: list,
            download_path):

        print(self.labeled_index)

        l_X_pool = self.X_train[self.labeled_index]
        l_y_pool = self.y_train[self.labeled_index]

        u_X_pool = np.delete(self.X_train, self.labeled_index, axis=0)
        u_y_pool = np.delete(self.y_train, self.labeled_index, axis=0)

        csv_path = os.path.join(download_path,
                                str(self.dataset_id),
                                f'{type(estimator).__name__}.csv')

        for idx in range(self.n_queries):

            u_pool_size = np.size(u_y_pool)

            if u_pool_size <= 0:
                warnings.warn(
                    f"(query {idx+1}/{self.n_queries}) Reserva de exemplos vazia.",
                    UserWarning)
                break

            # Extração de metafeatures dos dados rotulados
            mfs = self._extract_mfs(l_X_pool, l_y_pool)

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                query_index, score, strategy_name = self._topline_query(
                    estimator=estimator,
                    query_strategies=query_strategies,
                    l_pool=(l_X_pool, l_y_pool),
                    u_pool=(u_X_pool, u_y_pool),
                    batch_size=5,
                    committee_size=self.committee_size)

            mfs['dataset_id'] = int(self.dataset_id)
            mfs['query_number'] = idx
            mfs['estimator'] = type(estimator).__name__
            mfs['best_strategy'] = strategy_name
            mfs['best_score'] = score

            print(mfs)
            # Incluindo meta-exemplo na metabase
            mfs.to_frame().T.to_csv(csv_path, mode='a',
                                    header=(not os.path.exists(csv_path)))

            new_X, new_y = u_X_pool[query_index], u_y_pool[query_index]

            l_X_pool = np.append(l_X_pool, new_X, axis=0)
            l_y_pool = np.append(l_y_pool, new_y, axis=0)

            u_X_pool = np.delete(u_X_pool, query_index, axis=0)
            u_y_pool = np.delete(u_y_pool, query_index, axis=0)

if __name__ == '__main__':

    from sklearn.neighbors import KNeighborsClassifier
    from modAL.uncertainty import margin_sampling
    from modAL.disagreement import consensus_entropy_sampling
    from expected_error import expected_error_reduction
    from information_density import density_weighted

    pd.set_option('display.max_colwidth', None)

    query_strategies = [density_weighted,
                        margin_sampling,
                        consensus_entropy_sampling,
                        expected_error_reduction]

    builder = MetaBaseBuilder(
        dataset_id=40,
        initial_labeled_size=5,
        n_queries=3,
        batch_size=5,
        random_state=42)

    builder.run(estimator=KNeighborsClassifier(),
                download_path='.',
                query_strategies=query_strategies)


    
