import numpy as np
from functools import partial
import kerastuner as kt

from experiments.uci_income_census.utils import DataLoaderPreprocessorCensusUCI
from experiments.uci_income_census.hyper_mmoe import build_hyper_mmoe
from experiments.uci_income_census.tuner import IncomeCensusTuner

# random seed will define the split between validation and test set
SEED = 1

if __name__=="__main__":
    # load data and preprocess it
    data_preprocessor = DataLoaderPreprocessorCensusUCI(seed=SEED)

    # get data for the first group of tasks as defined in paper section 6.3:
    # Group 1, Task 1: predict whether income is above or below 50K
    # Group 2, Task 2: predict whether person was ever married
    (data_train_one, data_val_one, data_test_one), \
    [y_marital_train_one, y_marital_val_one, y_marital_test_one], \
    [y_income_train, y_income_val, y_income_test], \
    cat_features_dim = data_preprocessor.get_income_marital_stat_tasks()

    # define function that builds mmoe model based on the set of hyperparameters
    build_mmoe = partial(build_hyper_mmoe,
                         n_tasks=2,
                         all_columns=list(data_train_one.columns),
                         cat_features_dim=cat_features_dim,
                         val_data=(np.array(data_val_one.values,dtype=np.float32),
                                            y_marital_val_one,
                                            y_income_val),
                         output_layer_activation="sigmoid"
                        )

    # hyperp-parameter optimization
    hp_opt = kt.oracles.BayesianOptimization(objective=kt.Objective('auc', 'max'),
                                             max_trials=2)
    tuner = IncomeCensusTuner(oracle = hp_opt,
                              hypermodel=build_mmoe,
                              directory='results_hpopt_mmoe',
                              project_name='mmoe_uci_census_income')
    tuner.search(train_features = np.array(data_train_one.values, dtype=np.float32),
                 train_labels_main_task = np.array(y_income_train, dtype=np.float32),
                 train_labels_aux_task = np.array(y_marital_train_one, dtype=np.float32),
                 val_features = np.array(data_val_one.values, dtype=np.float32),
                 val_labels_main_task = np.array(y_income_val, dtype=np.float32))

