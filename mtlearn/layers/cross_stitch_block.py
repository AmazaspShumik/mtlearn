from typing import List

import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Layer


class CrossStitchBlock(Layer):
    """
    Cross-stitch block

    References
    ----------
    [1] Cross-stitch Networks for Multi-task Learning, (2017)
    Ishan Misra et al
    """

    def build(self,
              batch_input_shape: List[tf.TensorShape]
              ) -> None:
        stitch_shape = len(batch_input_shape)

        # initialize using random uniform distribution as suggested in ection 5.1
        self.cross_stitch_kernel = self.add_weight(shape=(stitch_shape, stitch_shape),
                                                   initializer=RandomUniform(0., 1.),
                                                   trainable=True,
                                                   name="cross_stitch_kernel")

        # normalize, so that each row will be convex linear combination,
        # here we follow recommendation in paper ( see section 5.1 )
        normalizer = tf.reduce_sum(self.cross_stitch_kernel,
                                   keepdims=True,
                                   axis=0)
        self.cross_stitch_unit.assign(self.cross_stitch_unit / normalizer)

    def call(self, inputs):
        """
        Forward pass through cross-stitch block

        Parameters
        ----------
        inputs: np.array or tf.Tensor
            List of task specific tensors
        """
        # vectorized cross-stitch unit operation
        x = tf.concat([tf.expand_dims(e, axis=-1) for e in inputs], axis=-1)
        stitched_output = tf.matmul(x, self.cross_stitch_unit)

        # split result into tensors corresponding to specific tasks and return
        # Note on implementation: applying tf.squeeze(*) on tensor of shape (None,x,1)
        # produces tensor of shape unknown, so we just extract 0-th element on last axis
        # through gather function
        outputs = [tf.gather(e, 0, axis=-1) for e in tf.split(stitched_output, len(inputs), axis=-1)]
        return outputs
