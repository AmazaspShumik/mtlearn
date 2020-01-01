#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class MLP(Layer):
    """
    Standard neural network with dense layers

    Parameters
    ----------
    architecture: List of ints
      Architecture of the network

    hidden_layer_activation: str
      Activation function for hidden layers

    output_layer_activation: str
      Activation function for the output layer

    layers: List of Layer instances
      List of layers (can be used in case you create layers outside and
      just want to make a compsite layer out of them). If provided then all other
      inputs are ignored.
    """

    def __init__(self,
                 architecture: List[int],
                 hidden_layer_activation: str = "elu",
                 output_layer_activation: str = None,
                 layers: Layer = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.layers = []
        if layers:
            self.layers = layers
        else:
            if not output_layer_activation:
                output_layer_activation = hidden_layer_activation
            # create hidden layers
            for hidden_units in architecture[:-1]:
                self.layers.append(Dense(hidden_units, hidden_layer_activation))
            # create output layer
            output_dim = architecture[-1]
            self.layers.append(Dense(output_dim, output_layer_activation))
        self.architecture = architecture
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation

    def call(self, inputs):
        """
        Forward pass of the MLP

        Parameters
        ----------
        inputs: np.array or tf.Tensor
          Input to the model

        Returns
        -------
        outputs: tf.Tensor
          Outputs of forward pass
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self) -> Dict:
        """ Get MLP configuration """
        base_config = super().get_config()
        return {**base_config,
                "layers": [tf.keras.layers.serialize(l) for l in self.layers],
                "architecture": self.architecture,
                "hidden_layer_activation": self.hidden_layer_activation,
                "output_layer_activation": self.output_layer_activation
                }
