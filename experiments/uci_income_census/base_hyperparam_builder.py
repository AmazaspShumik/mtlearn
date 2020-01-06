from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from kerastuner import HyperParameters
from mtlearn.layers import MLP
from experiments.uci_income_census import PreprocessingLayer
from typing import List, Dict


def build_activation_functions(hp: HyperParameters,
                               restricted_hyperparameter_search: bool):
    """ Helper method for setting activation functions """
    if restricted_hyperparameter_search:
        hp.Fixed("hidden_layer_activation","relu")
    else:
        hp.Choice("hidden_layer_activation", ["relu","elu","selu"])
    hp.Fixed("output_layer_activation","sigmoid")
    return hp


def build_task_towers(hp: HyperParameters,
                      n_tasks: int,
                      min_layers: int=1,
                      max_layers: int=5,
                      min_units_per_layer: int=8,
                      max_units_per_layer: int=40
                      ):
    """ Helper method to build task specific networks """
    task_towers = []
    n_layers = hp.Int(name="n_layers_tasks",
                      min_value=min_layers,
                      max_value=max_layers)
    for j in range(n_tasks):
        architecture = []
        for i in range(n_layers):
            n_units = hp.Int(name="n_units_layer_{0}_task_{1}".format(i, j),
                             min_value=min_units_per_layer,
                             max_value=max_units_per_layer)
            architecture.append(n_units)
        architecture.append(1)
        task_towers.append(MLP(architecture,
                               hp["hidden_layer_activation"],
                               hp["output_layer_activation"])
                          )
    return task_towers


def build_preprocessing_layer_uci_income(hp: HyperParameters,
                                         all_columns: List[str],
                                         cat_features_dim: Dict[str, int]
                                         ):
    """
    Helper method that builds preprocesing layer for UCI
    Census Income dataset.
    """
    feature_sparsity_threshold = hp.Int("feature_sparsity",
                                        min_value=3,
                                        max_value=10,
                                        default=3)
    return PreprocessingLayer(all_columns,
                              cat_features_dim,
                              feature_sparsity_threshold=feature_sparsity_threshold)

def build_experts(hp: HyperParameters):
    """ Helper method to build expert networks for OMOE and MMOE"""
    architecture = []
    n_experts = hp.Int("n_experts", 4, 10, default=6)
    n_layers = hp.Int("n_layers_experts", 2, 4, default=2)
    for i in range(n_layers):
        n_units = hp.Int("n_units_experts_{0}".format(i), 10, 20)
        architecture.append( n_units )
    return [MLP(architecture, hp["hidden_layer_activation"]) for _ in range(n_experts)]


