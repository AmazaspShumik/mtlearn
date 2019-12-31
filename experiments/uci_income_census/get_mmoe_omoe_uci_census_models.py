from functools import partial

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from experiments.uci_income_census_mmoe.preprocessing import PreprocessingLayer
from mtlearn.layers import MLP
from mtlearn.layers import MultiGateMixtureOfExperts, OneGateMixtureOfExperts


def get_moe_model_uci_census(all_columns,
                             cat_features_dim,
                             default_embed_size=10,
                             feature_sparsity_threshold=5,
                             n_experts=5,
                             expert_architecture=[32, 16],
                             task_specific_architecture=[8, 1],
                             hidden_layer_activation="selu",
                             output_layer_activation="sigmoid",
                             n_tasks=2,
                             moe_type="mmoe"
                             ):
    """ Create MMOE model for uci census income """
    # define experts: each expert is MLP
    get_expert = lambda: MLP(expert_architecture,
                             hidden_layer_activation,
                             hidden_layer_activation)
    experts = [get_expert() for _ in range(n_experts)]

    # define task specific layers: each task specific layer is MLP
    get_task_mlp = lambda: MLP(task_specific_architecture,
                               hidden_layer_activation,
                               output_layer_activation)
    task_towers = [get_task_mlp() for _ in range(n_tasks)]

    # adding preprocessing layer that converts categorical
    # columns either to one-hot encoding or embedding layer
    base_layer = PreprocessingLayer(all_columns,
                                    cat_features_dim,
                                    default_embed_size=default_embed_size,
                                    feature_sparsity_threshold=feature_sparsity_threshold)

    # combine everything using Multi Gate Mixture of Experts layer
    input_layer = Input(shape=(len(all_columns),))
    if moe_type == 'mmoe':
        output_layer = MultiGateMixtureOfExperts(experts,
                                                 task_towers,
                                                 moe_dropout=False,
                                                 moe_dropout_rate=0.,
                                                 base_layer=base_layer)(input_layer)
    else:
        output_layer = OneGateMixtureOfExperts(experts,
                                               task_towers,
                                               moe_dropout=False,
                                               moe_dropout_rate=0.,
                                               base_layer=base_layer)(input_layer)
    return Model(inputs=input_layer, outputs=output_layer)


get_mmoe_uci_census = partial(get_moe_model_uci_census, moe_type="mmoe")
get_omoe_uci_census = partial(get_moe_model_uci_census, moe_type="omoe")
