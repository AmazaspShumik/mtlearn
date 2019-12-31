from unittest import TestCase

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from mtlearn.layers import MultiGateMixtureOfExperts
from tests import BaseSpecificationMTL


def get_mmoe_model(input_shape: int,
                   apply_dropout: bool,
                   dropout_rate: float = 0.,
                   n_experts: int = 10,
                   activation_fn: str = "selu",
                   include_base_layer: bool = False,
                   n_tasks: int = 2):
    """
    Create Multigate Mixture of Experts Model

    Parameters
    ----------
    input_shape: int
        Number of features

    apply_dropout: bool
        Apply dropout

    dropout_rate: float
        Probability with which neurons will be randomly
        omitted during training.

    n_experts: int (default=10)
        Number of experts

    include_base_layer: bool
        Include base layer ( inputs to every task use the
        same base layer)

    n_tasks: int (default=2)
        Number of tasks ( in muli-task learning problem)

    Returns
    -------
    model: tf.keras.models.Model
        Compiled model
    """
    # define experts: each expert is dense layer with user defined
    # activation function
    experts = [Dense(10, activation_fn) for _ in range(n_experts)]

    # task layers: each task layer is a dense layer with user defined
    # activation function
    task_layers = [Dense(1, activation_fn) for _ in range(n_tasks)]

    # adding base / preprocessing layer
    base_layer = None
    if include_base_layer:
        base_layer = Dense(input_shape, 'linear')

    # combine everything using Multi Gate Mixture of Experts layer
    mmoe = MultiGateMixtureOfExperts(experts,
                                     task_layers,
                                     moe_dropout=apply_dropout,
                                     moe_dropout_rate=dropout_rate,
                                     base_layer=base_layer)

    # create tf.keras Model using functional API
    input_layer = Input(shape=(input_shape,))
    output_layer = mmoe(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mse", optimizer="adam")
    return model


class TestMultiGateMixtureOfExperts(TestCase, BaseSpecificationMTL):
    """ Unit tests for Muli-Gate Mixture of Experts Layer """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_data_and_directory()

    @property
    def models(self):
        """
        Create MMOEs models for testing
        """
        return [get_mmoe_model(1, True, 0.5),
                get_mmoe_model(1, False),
                get_mmoe_model(1, False, include_base_layer=True)]
