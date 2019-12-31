from unittest import TestCase

from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from mtlearn.layers import ConstrainedMTL
from tests import BaseSpecificationMTL


def get_constrained_model(n_features,
                          l1_regularizer,
                          l2_regularizer
                          ):
    """
    Builds soft parameter sharing multi-task model

    n_features: int
       Number of features in the data

    l1_regularizer: float
       L1 penalty strength

    l2_regularizer: float
       L2 penalty strength

    Returns
    -------
    model: instance of tf.keras.models.Model
       Compiled MTL model
    """

    # define all layers
    input_layer = Input(shape=(n_features,))
    first_shared_layer = [Dense(8, "selu") for _ in range(2)]
    second_shared_layer = [Dense(4, "selu") for _ in range(2)]
    last_shared_layer = [Dense(1, "selu") for _ in range(2)]

    # build forward propagation for soft parameter sharing mtl
    layer_1 = ConstrainedMTL(first_shared_layer, l1_regularizer, l2_regularizer)(input_layer)
    layer_2 = ConstrainedMTL(second_shared_layer, l1_regularizer, l2_regularizer)(layer_1)
    output_layer = ConstrainedMTL(last_shared_layer, l1_regularizer, l2_regularizer)(layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mse", optimizer="adam")
    return model


class TestConstrainedMTL(TestCase, BaseSpecificationMTL):
    """ Test Multi-task learning through soft parameter sharing """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_data_and_directory()

    def test_input_types_output_shapes(self) -> None:
        """
        Checks whether layer can properly handle two types of input:
         1) list of tensors
         2) tensors
        And whether output shapes are a they are expected.
        """
        input_tensor = tf.random.uniform([5,3])
        input_list_tensor = [tf.random.uniform([5,3]) for _ in range(2)]
        mtl_layers = [Dense(8, "selu") for _ in range(2)]
        constraining_layer = ConstrainedMTL(mtl_layers, 1., 1.)
        output_tensor = constraining_layer(input_tensor, training=False)
        output_list_tensor = constraining_layer(input_list_tensor, training=False)
        self.assertTrue(output_tensor[0].shape == tf.TensorShape([5,8]))
        self.assertTrue(output_tensor[1].shape == tf.TensorShape([5,8]))
        self.assertTrue(output_list_tensor[0].shape == tf.TensorShape([5,8]))
        self.assertTrue(output_list_tensor[1].shape == tf.TensorShape([5,8]))

    @property
    def models(self) -> List[Model]:
        return [get_constrained_model(1, 0.5, 0.5),
                get_constrained_model(1, 0., 1.),
                get_constrained_model(1, 1., 0.)
                ]
