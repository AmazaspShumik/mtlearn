from typing import List, Dict, Tuple

import tensorflow as tf
from kerastuner import HyperParameters
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from experiments.uci_income_census.base_hyperparam_builder import (build_task_towers, build_optimizer,
                                                                   build_experts, build_preprocessing_layer_uci_income)
from mtlearn.layers import MultiGateMixtureOfExperts


def build_hyper_mmoe(hp: HyperParameters,
                     n_tasks: int,
                     all_columns: List[str],
                     cat_features_dim: Dict[str, int],
                     val_data: Tuple,
                     output_layer_activation: str):
    """
    Build Multi-Gate Mixture of Experts

    Parameters
    ----------
    hp: instance of HyperParameters
        Hyper-Parameters that define architecture and training of neural networks

    Returns
    -------

    """
    hidden_layer_activation = hp.Choice("hidden_layer_activation", ["elu", "relu", "selu"])
    output_layer_activation = hp.Fixed("output_layer_activation", output_layer_activation)
    experts = build_experts(hp)
    task_towers = build_task_towers(hp, n_tasks)
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim)
    mmoe = MultiGateMixtureOfExperts(experts,
                                     task_towers,
                                     base_layer=preprocessing_layer)
    input_layer = Input(shape=(len(all_columns),))
    output_layer = mmoe(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                  optimizer=build_optimizer(hp),
                  validation_data=val_data,
                  metrics=[tf.keras.metrics.AUC()]  # , tf.keras.metrics.AUC()]
                  )
    return model
