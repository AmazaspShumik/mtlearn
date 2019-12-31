from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model, load_model

from experiments.uci_income_census_mmoe.data_loader_preprocessor import DataLoaderPreprocessorCensusUCI

SEED = 1


def evaluate(test_first_target: np.array,
             test_second_target: np.array,
             features: np.array,
             model: Model
             ) -> Tuple[float]:
    """ Evaluate AUC for model with two tasks """
    y_pred_first, y_pred_second = model.predict(features)
    auc_first = roc_auc_score(test_first_target, y_pred_first)
    auc_second = roc_auc_score(test_second_target, y_pred_second)
    return auc_first, auc_second


# load data and preprocess data
data_preprocessor = DataLoaderPreprocessorCensusUCI(seed=SEED)

# ====================== Group One of tasks ===================================

# As defined in the paper group one of tasks consists of two tasks:
# Group 1, Task 1: predict whether income is above or below 50K
# Group 2, Task 2: predict whether person was ever married

# get data
(data_train_one, data_val_one, data_test_one), \
[y_marital_train_one, y_marital_val_one, y_marital_test_one], \
[y_income_train, y_income_val, y_income_test], \
cat_features_dim = data_preprocessor.get_income_marital_stat_tasks()
features_group_one_test = np.array(data_test_one.values, dtype=np.float32)
features_group_one_val = np.array(data_val_one.values, dtype=np.float32)

# evaluate with MMOE
mmoe_model = load_model("mmoe_marital_income")
omoe_model = load_model("omoe_marital_income")
models = [mmoe_model, omoe_model]
model_names = ["Multi-Gate Mixture of Experts",
               "One-Gate Mixture of Experts"]

print(" \n Test set results")
for model, model_name in zip(models, model_names):
    group_one_auc_marital_test, group_one_auc_income_test = evaluate(y_marital_test_one,
                                                                     y_income_test,
                                                                     features_group_one_test,
                                                                     model
                                                                     )
    print("\n Model - {0}".format(model_name))
    print("   Group 1, AUC marital task test set: {0}".format(group_one_auc_marital_test))
    print("   Group 1, AUC income task test set: {0}".format(group_one_auc_income_test))

print(" \n Validation set results")
for model, model_name in zip(models, model_names):
    group_one_auc_marital_val, group_one_auc_income_val = evaluate(y_marital_val_one,
                                                                   y_income_val,
                                                                   features_group_one_val,
                                                                   model
                                                                   )
    print("\n Model - {0}".format(model_name))
    print("  Group 1, AUC marital task validation set: {0}".format(group_one_auc_marital_val))
    print("  Group 1, AUC income task validation set: {0}".format(group_one_auc_income_val))

print("Multigate Mixture of Experts, results: ")
