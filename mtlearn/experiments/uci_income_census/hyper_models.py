from functools import partial
from typing import List, Dict

from kerastuner import HyperParameters
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from mtlearn.experiments import (build_task_towers,
                                 build_activation_functions,
                                 build_experts,
                                 build_preprocessing_layer_uci_income,
                                 )
from mtlearn.layers import (MultiGateMixtureOfExperts, OneGateMixtureOfExperts,
                            CrossStitchBlock, ConstrainedMTL, MLP)


def build_hyper_moe_model(hp: HyperParameters,
                          n_tasks: int,
                          all_columns: List[str],
                          cat_features_dim: Dict[str, int],
                          restricted_hyperparameter_search: bool,
                          min_experts: int = 4,
                          max_experts: int = 16,
                          min_layers_per_expert: int = 2,
                          max_layers_per_expert: int = 3,
                          min_units_per_layer_expert: int = 32,
                          max_units_per_layer_expert: int = 64,
                          min_task_layers: int = 1,
                          max_task_layers: int = 3,
                          min_units_per_task_layer: int = 8,
                          max_units_per_task_layer: int = 16,
                          feature_sparsity_min: int = 4,
                          feature_sparsity_max: int = 9,
                          moe_type: str = "mmoe"
                          ) -> Model:
    """
    Build hypermodel either for Multi-Gate Mixture of Experts or
    One-Gate Mixture of experts

    Parameters
    ----------
    hp: instance of HyperParameters
        Hyper-Parameters that define architecture and training of neural networks

    n_tasks: int
        Number of tasks

    all_columns: list
        Names of the features

    cat_features_dim: dict
        Dictionary that maps from the name of categorical feature
        to its dimensionality

    restricted_hyperparameter_search: bool
        If True, then fixes following hyperparameters and does not optimize them.
        - batch_size = 1024
        - hidden_layer_activation = relu
        - optimizer = sgd

    min_experts: int
        Minimum number of experts

    max_experts: int
        Maximum number of experts

    min_layers_per_expert: int
        Minimum number of layers for each expert

    max_layers_per_expert: int
        Maximum number of layers for each expert

    min_units_per_layer_expert:
        Minimum number of units for each layer of the expert

    max_units_per_layer_expert:
        Maximum number of units for each layer of the expert

    min_task_layers: int
        Minimum number of layers in each task tower

    max_task_layers: int
        Maximum number of layers in each task tower

    min_units_per_task_layer: int
        Minimum number of units in each layer of task tower

    max_units_per_layer: int
        Maximum number of units in each layer of task tower

    moe_type: str (either "mmoe" or "omoe")
        Type of Mixture of Experts Model, there are two possibilities
        Multi-Gate Mixture of Experts and One-Gate Mixture of Experts.

    feature_sparsity_min: int
        Minimum possible value of feature sparsity threshold

    feature_sparsity_max: int
        Maximum possible value of feature sparsity threshold

    moe_type: str
        Either "mmoe" for Multi-Gate Mixture of Experts or "omoe"
        for One-Gate Mixture of Experts

    Returns
    -------
    model: tensorflow.keras.models.Model
        Compiled MMOE or OMOE model
    """
    build_activation_functions(hp, restricted_hyperparameter_search)
    experts = build_experts(hp,
                            min_experts,
                            max_experts,
                            min_layers_per_expert,
                            max_layers_per_expert,
                            min_units_per_layer_expert,
                            max_units_per_layer_expert)
    task_towers = build_task_towers(hp,
                                    n_tasks,
                                    min_task_layers,
                                    max_task_layers,
                                    min_units_per_task_layer,
                                    max_units_per_task_layer
                                    )
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim,
                                                               feature_sparsity_min,
                                                               feature_sparsity_max
                                                               )
    if moe_type == "mmoe":
        top_layer = MultiGateMixtureOfExperts(experts,
                                              task_towers,
                                              base_layer=preprocessing_layer)
    else:
        top_layer = OneGateMixtureOfExperts(experts,
                                            task_towers,
                                            base_layer=preprocessing_layer)
    input_layer = Input(shape=(len(all_columns),))
    output_layer = top_layer(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def build_hyper_cross_stitched_model(hp: HyperParameters,
                                     n_tasks: int,
                                     all_columns: List[str],
                                     cat_features_dim: Dict[str, int],
                                     restricted_hyperparameter_search: bool,
                                     feature_sparsity_min: int = 4,
                                     feature_sparsity_max: int = 9,
                                     min_layers: int = 3,
                                     max_layers: int = 6,
                                     min_units_per_layer: int = 32,
                                     max_units_per_layer: int = 64
                                     ) -> Model:
    """
    Build model for Cross Stitched networks

    Parameters
    ----------
    hp: instance of HyperParameters
        Hyper-Parameters that define architecture and training of neural networks

    n_tasks: int
        Number of tasks

    all_columns: list
        Names of the features

    cat_features_dim: dict
        Dictionary that maps from the name of categorical feature
        to its dimensionality.

    restricted_hyperparameter_search: bool
        If True, then fixes following hyperparameters and does not optimize them.
        - batch_size = 1024
        - hidden_layer_activation = relu
        - optimizer = sgd

    feature_sparsity_min: int
        Minimum possible value of feature sparsity threshold

    feature_sparsity_max: int
        Maximum possible value of feature sparsity threshold

    min_layers: int
        Minimum number of layers

    max_layers: int
        Maximum number of layers

    min_units_per_layer: int
        Minimum number of neurons per layer

    max_units_per_layer: int
        Maximum number of neurons per layer

    Returns
    -------
    model: tensorflow.keras.models.Model
        Compiled Cross Stitched Networks Model
    """
    # define activation functions and preproceing layer
    build_activation_functions(hp, restricted_hyperparameter_search)
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim,
                                                               feature_sparsity_min,
                                                               feature_sparsity_max)
    # propagate input through preprocesing layer
    input_layer = Input(shape=(len(all_columns),))
    x = preprocessing_layer(input_layer)

    # build cross-stitch network model
    n_layers = hp.Int("number_of_hidden_layers",
                      min_value=min_layers,
                      max_value=max_layers)
    for i in range(n_layers):
        n_units = hp.Int("n_units_layer_{0}".format(i),
                         min_value=min_units_per_layer,
                         max_value=max_units_per_layer)
        dense_layers_output = [Dense(n_units, hp["hidden_layer_activation"])(x) for _ in range(n_tasks)]
        x = CrossStitchBlock()(dense_layers_output)
    output_layers = [Dense(1, hp['output_layer_activation'])(x) for _ in range(n_tasks)]
    model = Model(inputs=input_layer, outputs=output_layers)
    return model


def build_hyper_l2_constrained(hp: HyperParameters,
                               n_tasks: int,
                               all_columns: List[str],
                               cat_features_dim: Dict[str, int],
                               restricted_hyperparameter_search: bool,
                               feature_sparsity_min: int = 4,
                               feature_sparsity_max: int = 9,
                               min_layers: int = 3,
                               max_layers: int = 6,
                               min_units_per_layer: int = 32,
                               max_units_per_layer: int = 64,
                               min_l2_alpha: float = 1e-1,
                               max_l2_alpha: float = 1e+2
                               ) -> Model:
    """
    Build model for L2 constrained multi-task learning model

    Parameters
    ----------
    hp: instance of HyperParameters
        Hyper-Parameters that define architecture and training of neural networks

    n_tasks: int
        Number of tasks

    all_columns: list
        Names of the features

    cat_features_dim: dict
        Dictionary that maps from the name of categorical feature
        to its dimensionality.

    restricted_hyperparameter_search: bool
        If True, then fixes following hyperparameters and does not optimize them.
        - batch_size = 1024
        - hidden_layer_activation = relu
        - optimizer = sgd

    feature_sparsity_min: int
        Minimum possible value of feature sparsity threshold

    feature_sparsity_max: int
        Maximum possible value of feature sparsity threshold

    min_layers: int
        Minimum number of layers

    max_layers: int
        Maximum number of layers

    min_units_per_layer: int
        Minimum number of neurons per layer

    max_units_per_layer: int
        Maximum number of neurons per layer

    min_l2_alpha: float
        Minimum possible value of l2 regularization coefficient

    max_l2_alpha: float
        Maximium possible value of l2 regularization coefficient

    Returns
    -------
    model: tensorflow.keras.models.Model
        Compiled L2 Constrained Model
    """
    # define activation functions and preproceing layer
    build_activation_functions(hp, restricted_hyperparameter_search)
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim,
                                                               feature_sparsity_min,
                                                               feature_sparsity_max)
    # propagate input through preprocesing layer
    input_layer = Input(shape=(len(all_columns),))
    x = preprocessing_layer(input_layer)

    # build l2 constrained model
    n_layers = hp.Int("number_of_hidden_layers",
                      min_value=min_layers,
                      max_value=max_layers)
    for i in range(n_layers):
        n_units = hp.Int("n_units_layer_{0}".format(i),
                         min_value=min_units_per_layer,
                         max_value=max_units_per_layer)
        mtl_layers = [Dense(n_units, hp['hidden_layer_activation']) for _ in range(n_tasks)]
        l2_regularizer = hp.Float("l2_regularizer_layer_{0}".format(i),
                                  min_value=min_l2_alpha,
                                  max_value=max_l2_alpha)
        constrained_l2 = ConstrainedMTL(mtl_layers, l1_regularizer=0., l2_regualrizer=l2_regularizer)
        x = constrained_l2(x)
    output_layers = [Dense(1, hp['output_layer_activation'])(x[i]) for i in range(n_tasks)]
    model = Model(inputs=input_layer, outputs=output_layers)
    return model


def build_hyper_mtl_shared_bottom(hp: HyperParameters,
                                  n_tasks: int,
                                  all_columns: List[str],
                                  cat_features_dim: Dict[str, int],
                                  restricted_hyperparameter_search: bool,
                                  min_shared_layers: int = 2,
                                  max_shared_layers: int = 3,
                                  min_task_layers: int = 1,
                                  max_task_layers: int = 3,
                                  min_units_per_layer_shared: int = 32,
                                  max_units_per_layer_shared: int = 64,
                                  min_units_per_layer_task: int = 8,
                                  max_units_per_layer_task: int = 16
                                  ):
    """
    Build model for L2 constrained multi-task learning model

    Parameters
    ----------
    hp: instance of HyperParameters
        Hyper-Parameters that define architecture and training of neural networks

    n_tasks: int
        Number of tasks

    all_columns: list
        Names of the features

    cat_features_dim: dict
        Dictionary that maps from the name of categorical feature
        to its dimensionality.

    restricted_hyperparameter_search: bool
        If True, then fixes following hyperparameters and does not optimize them.
        - batch_size = 1024
        - hidden_layer_activation = relu
        - optimizer = sgd

    Returns
    -------
    model: tensorflow.keras.models.Model
        Compiled standard MTL model with hard parameter sharing
    """
    # define activation functions and preproceing layer
    build_activation_functions(hp, restricted_hyperparameter_search)
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim)
    # propagate input through preprocesing layer
    input_layer = Input(shape=(len(all_columns),))
    x = preprocessing_layer(input_layer)

    # build shared layers
    architecture = []
    n_layers = hp.Int("n_layers_experts",
                      min_shared_layers,
                      max_shared_layers)
    for i in range(n_layers):
        n_units = hp.Int("n_units_experts_{0}".format(i),
                         min_units_per_layer_shared,
                         max_units_per_layer_task)
        architecture.append(n_units)
    shared_layers = MLP(architecture, hp["hidden_layer_activation"])
    shared_layers_output = shared_layers(input_layer)

    # task layers
    task_towers = build_task_towers(hp,
                                    n_tasks,
                                    min_task_layers,
                                    max_task_layers,
                                    min_units_per_layer_task,
                                    max_units_per_layer_task)
    output_layer = [task(shared_layers_output) for task in task_towers]
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


build_hyper_mmoe_model = partial(build_hyper_moe_model, moe_type="mmoe")

build_hyper_omoe_model = partial(build_hyper_moe_model, moe_type="omoe")
