import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score

from experiments.uci_income_census_mmoe.data_loader_preprocessor import DataLoaderPreprocessorCensusUCI
from experiments.uci_income_census_mmoe import get_mmoe_uci_census, get_omoe_uci_census

# random seed will define the split between validation and test set
SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)


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

    # create omoe model, compile and fit it
    omoe_model = get_omoe_uci_census( list(data_train_one.columns),
                                      cat_features_dim,
                                      hidden_layer_activation="relu"
                                    )
    omoe_model.compile( loss=['binary_crossentropy', 'binary_crossentropy'],
                        optimizer="adam",
                        validation_data=(np.array(data_val_one.values,dtype=np.float32),
                                         y_marital_val_one,
                                         y_income_val),
                        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC()]
                      )
    omoe_model.fit(x=np.array(data_train_one.values, dtype=np.float32),
                   y=[y_marital_train_one, y_income_train],
                   epochs=25,
                   batch_size=1024)

    # create mmoe model, compile and fit it
    mmoe_model = get_mmoe_uci_census( list(data_train_one.columns),
                                      cat_features_dim,
                                      hidden_layer_activation = "relu"
                                    )
    mmoe_model.compile( loss=['binary_crossentropy', 'binary_crossentropy'],
                        optimizer = "adam",
                        validation_data=(np.array(data_val_one.values,dtype=np.float32),
                                         y_marital_val_one,
                                         y_income_val),
                        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC()]
                      )
    mmoe_model.fit(x=np.array(data_train_one.values, dtype=np.float32),
                   y=[y_marital_train_one, y_income_train],
                   epochs=25,
                   batch_size=1024)

    mmoe_model.save("mmoe_marital_income")
    omoe_model.save("omoe_marital_income")