from typing import List, Dict

from kerastuner import HyperParameters
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from experiments.uci_income_census.base_hyperparam_builder import (build_task_towers,
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
                          moe_type: str,
                          restricted_hyperparameter_search: bool):
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

    moe_type: str (either "mmoe" or "omoe")
        Type of Mixture of Experts Model, there are two possibilities
        Multi-Gate Mixture of Experts and One-Gate Mixture of Experts.

    restricted_hyperparameter_search: bool
        If True, then fixes following hyperparameters and does not optimize them.
        - batch_size = 1024
        - hidden_layer_activation = relu
        - optimizer = sgd

    Returns
    -------
    model: tensorflow.keras.models.Model
        Compiled MMOE or OMOE model
    """
    build_activation_functions(hp, restricted_hyperparameter_search)
    experts = build_experts(hp)
    task_towers = build_task_towers(hp, n_tasks)
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim)
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
                                     restricted_hyperparameter_search: bool
                                     ):
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

    Returns
    -------
    model: tensorflow.keras.models.Model
        Compiled Cross Stitched Networks Model
    """
    # define activation functions and preproceing layer
    build_activation_functions(hp, restricted_hyperparameter_search)
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim)
    # propagate input through preprocesing layer
    input_layer = Input(shape=(len(all_columns),))
    x = preprocessing_layer(input_layer)

    # build cross-stitch network model
    n_layers = hp.Int("number_of_hidden_layers", min_value=2, max_value=8)
    for i in range(n_layers):
        n_units = hp.Int("n_units_layer_{0}".format(i), min_value=8, max_value=40)
        dense_layers_output = [Dense(n_units, hp["hidden_layer_activation"])(x) for _ in range(n_tasks)]
        x = CrossStitchBlock()(dense_layers_output)
    output_layers = [Dense(1, hp['output_layer_activation'])(x) for _ in range(n_tasks)]
    model = Model(inputs=input_layer, outputs=output_layers)
    return model


def build_hyper_l2_constrained(hp: HyperParameters,
                               n_tasks: int,
                               all_columns: List[str],
                               cat_features_dim: Dict[str, int],
                               restricted_hyperparameter_search: bool):
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
        Compiled L2 Constrained Model
    """
    # define activation functions and preproceing layer
    build_activation_functions(hp, restricted_hyperparameter_search)
    preprocessing_layer = build_preprocessing_layer_uci_income(hp,
                                                               all_columns,
                                                               cat_features_dim)
    # propagate input through preprocesing layer
    input_layer = Input(shape=(len(all_columns),))
    x = preprocessing_layer(input_layer)

    # build l2 constrained model
    n_layers = hp.Int("number_of_hidden_layers", min_value=2, max_value=8)
    for i in range(n_layers):
        n_units = hp.Int("n_units_layer_{0}".format(i), min_value=8, max_value=40)
        mtl_layers = [Dense(n_units, hp['hidden_layer_activation']) for _ in range(n_tasks)]
        l2_regularizer = hp.Float("l2_regularizer_layer_{0}".format(i), min_value=1e-1, max_value=1e+2)
        constrained_l2 = ConstrainedMTL(mtl_layers, l1_regularizer=0., l2_regualrizer=l2_regularizer)
        x = constrained_l2(x)
    output_layers = [Dense(1, hp['output_layer_activation'])(x[i]) for i in range(n_tasks)]
    model = Model(inputs=input_layer, outputs=output_layers)
    return model


def build_mtl_shared_bottom(hp: HyperParameters,
                            n_tasks: int,
                            all_columns: List[str],
                            cat_features_dim: Dict[str, int],
                            restricted_hyperparameter_search: bool):
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
    n_layers = hp.Int("n_layers_experts", 2, 4, default=2)
    for i in range(n_layers):
        n_units = hp.Int("n_units_experts_{0}".format(i), 10, 20)
        architecture.append( n_units )
    shared_layers = MLP(architecture, hp["hidden_layer_activation"])
    shared_layers_output = shared_layers(input_layer)

    # task layers
    task_towers = build_task_towers(hp, n_tasks)
    output_layer = [task(shared_layers_output) for task in task_towers]
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

