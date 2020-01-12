from typing import List, Dict

from kerastuner import HyperParameters

from mtlearn.experiments import PreprocessingLayer
from mtlearn.layers import MLP


def build_activation_functions(hp: HyperParameters,
                               restricted_hyperparameter_search: bool):
    """ Helper method for setting activation functions """
    if restricted_hyperparameter_search:
        hp.Fixed("hidden_layer_activation", "relu")
    else:
        hp.Choice("hidden_layer_activation", ["relu", "elu", "selu"])
    hp.Fixed("output_layer_activation", "sigmoid")
    return hp


def build_task_towers(hp: HyperParameters,
                      n_tasks: int,
                      min_layers: int = 1,
                      max_layers: int = 3,
                      min_units_per_layer: int = 8,
                      max_units_per_layer: int = 16
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
                                         cat_features_dim: Dict[str, int],
                                         feature_sparsity_min: int = 3,
                                         feature_sparsity_max: int = 8
                                         ):
    """
    Helper method that builds preprocesing layer for UCI
    Census Income dataset.
    """
    feature_sparsity_threshold = hp.Int("feature_sparsity",
                                        min_value=feature_sparsity_min,
                                        max_value=feature_sparsity_max,
                                        default=3)
    return PreprocessingLayer(all_columns,
                              cat_features_dim,
                              feature_sparsity_threshold=feature_sparsity_threshold)


def build_experts(hp: HyperParameters,
                  min_experts: int,
                  max_experts: int,
                  min_layers_per_expert: int,
                  max_layers_per_expert: int,
                  min_units_per_layer: int,
                  max_units_per_layer: int):
    """ Helper method to build expert networks for OMOE and MMOE"""
    architecture = []
    n_experts = hp.Int("n_experts", min_experts, max_experts, default=6)
    n_layers = hp.Int("n_layers_experts", min_layers_per_expert, max_layers_per_expert, default=2)
    for i in range(n_layers):
        n_units = hp.Int("n_units_experts_{0}".format(i),
                         min_units_per_layer,
                         max_units_per_layer)
        architecture.append(n_units)
    return [MLP(architecture, hp["hidden_layer_activation"]) for _ in range(n_experts)]
