from unittest import TestCase

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from mtlearn.layers import CrossStitchBlock
from tests import BaseSpecificationMTL


def get_cross_stitch_model(n_features: int):
    """
    Creates model with cross-stitch units

    Parameters
    ----------
    n_features: int
        Shape of the input
    """
    input_layer = Input(shape=(n_features,))
    layer_1 = [Dense(12, "selu")(input_layer), Dense(12, "selu")(input_layer)]
    cs_block_1 = CrossStitchBlock()(layer_1)
    layer_2 = [Dense(6, "selu")(stitch) for stitch in cs_block_1]
    cs_block_2 = CrossStitchBlock()(layer_2)
    output_layers = [Dense(1, "selu")(stitch) for stitch in cs_block_2]
    model = Model(inputs=input_layer, outputs=output_layers)
    model.compile(loss="mse", optimizer="adam")
    return model


class TestCrossStitch(TestCase, BaseSpecificationMTL):
    """ Test cross-stitch block """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_data_and_directory()

    @property
    def models(self):
        return [get_cross_stitch_model(1)]

    def test_cross_stitch_normalization_and_shape(self):
        """
        Test that elements of each column are convex combinations,
        and that output is a list of tensors with the same dimensionality
        as
        """
        test_input = [tf.ones([2, 3], dtype=tf.float32) for _ in range(4)]
        cross_stitch = CrossStitchBlock()
        output = cross_stitch(test_input)

        # check column normalization
        col_sum = tf.reduce_sum(cross_stitch.cross_stitch_kernel, 0)
        expected_col_sum = tf.ones(4, dtype=tf.float32)
        abs_diff = tf.abs(col_sum - expected_col_sum)
        self.assertTrue(tf.reduce_sum(abs_diff) < 1e-4)

        # check shape
        for i in range(4):
            self.assertTrue(tf.reduce_all(output[i].shape == tf.TensorShape([2, 3])))
