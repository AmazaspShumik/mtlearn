#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Dict, List, Sequence

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

from mtlearn.utils import has_arg


class ExpertUtilizationDropout(Layer):
    """
    Helper layer for Mixture of Experts Layer, allows to drop some of the experts
    and then renormalize other experts, so that we still have probability density
    function for expert utilization.

    Parameters
    ----------
    dropout_rate: float (between 0 and 1), optional (Default=0.1)
      Probability of drop

    References
    ----------
    [1] Recommending What Video to Watch Next: A Multitask Ranking System, 2019.
        Zhe Zhao
    """

    def __init__(self,
                 dropout_rate: float = 0.1,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.dropout_rate = dropout_rate

    def build(self,
              batch_input_shape: Tuple[int]
              ) -> None:
        """
        Builds main component of the layer by creating dropout layer that has
        the same binary mask for each element in the batch.

        Parameters
        ----------
        batch_input_shape: tuple
          Shape of the inputs
        """
        self.drop_layer = Dropout(self.dropout_rate,
                                  noise_shape=(1, *batch_input_shape[1:]),
                                  **self.kwargs
                                  )
        super().build(batch_input_shape)

    def call(self,
             inputs,
             training
             ):
        """
        Defines forward pass for the layer, by dropping experts and then
        renormalizing utilization probabilities for remaining experts.

        Parameters
        ----------
        inputs: np.array or tf.Tensor
          Inputs to the layer (can be numpy array, tf.Tensor)

        training: bool
          True if layer is called in training mode, False otherwise

        Returns
        -------
        : tf.Tensor
          Renormalized probabilities assigned to each expert
        """
        expert_probs_drop = self.drop_layer(inputs, training)
        # add epsilon to prevent division by zero when all experts are dropped
        normalizer = tf.add(tf.reduce_sum(expert_probs_drop, keepdims=True, axis=1),
                            tf.keras.backend.epsilon()
                            )
        return expert_probs_drop / normalizer

    def get_config(self) -> Dict:
        """ Config of the ExpertUtilizationDropout layer """
        base_config = super().get_config()
        return {**base_config, "dropout_rate": self.dropout_rate}


class MixtureOfExpertsLayer(Layer):
    """
    Mixture of Experts Layer

    Parameters
    ----------
    expert_layers: List of Layers
        List of experts, each expert is expected to be a instance of Layer class.

    add_dropout: bool, optional (Default=False)
        Adds dropout after softmax layer, helps to avoid collapse of the experts
        and their imbalanced utilization. Was used by Zhe Zhao in [1].

    dropout_rate: float (between 0 and 1), optional (Default=0.1)
        Fraction of the experts to drop.

    base_expert_prob_layer: tf.keras.Layer, optional (Default=None)
        Layer that extracts features from inputs.

    References
    ----------
    [1] Recommending What Video to Watch Next: A Multitask Ranking System, 2019.
        Zhe Zhao
    [2] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
        Layer, 2017. Noam Shazeer, Azalia Mirhoseini.
    """

    def __init__(self,
                 expert_layers: List[Layer],
                 add_dropout: bool = False,
                 dropout_rate: float = 0.1,
                 base_expert_prob_layer: Layer = None,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.expert_layers = expert_layers
        self.n_experts = len(expert_layers)
        self.add_dropout = add_dropout
        self.dropout_rate = dropout_rate
        self.base_expert_prob_layer = base_expert_prob_layer
        self.expert_probs = Dense(self.n_experts, "softmax", **self.kwargs)
        if add_dropout:
            self.drop_expert_layer = ExpertUtilizationDropout(self.dropout_rate)

    def call(self, inputs, training):
        """
        Defines set of computations performed in the MOE layer.
        MOE layer can accept single tensor (in this case it assumes the same
        input for every expert) or collection/sequence of tensors
        (in this case it assumes every tensor corresponds to its own expert)

        Parameters
        ----------
        inputs: np.array, tf.Tensor, list/tuple of np.arrays or  tf.Tensors
          Inputs to the MOE layer

        training: bool
          True if layer is called in training mode, False otherwise

        Returns
        -------
        moe_output: tf.Tensor
          Output of mixture of experts layers ( linearly weighted output of expert
          layers).
        """
        # compute each expert output (optionally pass training argument,
        # since some experts may contain training arg, some may not.
        experts_output = []
        for expert in self.expert_layers:
            if has_arg(expert, "training"):
                experts_output.append(expert(inputs, training))
            else:
                experts_output.append(expert(inputs))

        # compute probability of expert (degree of expert utilization) for given
        # input set
        if self.base_expert_prob_layer:
            inputs = self.base_expert_prob_layer(inputs)
        expert_utilization_prob = self.expert_probs(inputs)
        if self.add_dropout:
            expert_utilization_prob = self.drop_expert_layer(expert_utilization_prob,
                                                             training)

        # compute weighted output of experts
        moe_output = 0
        for i, expert_output in enumerate(experts_output):
            moe_output += (expert_output
                           * tf.expand_dims(expert_utilization_prob[:, i], axis=-1)
                           )
        return moe_output

    def get_config(self) -> Dict:
        """ Config of the MOE layer """
        base_config = super().get_config()
        return {**base_config,
                "add_dropout": self.add_dropout,
                "dropout_rate": self.dropout_rate,
                "expert_layers": [tf.keras.layers.serialize(layer) for
                                  layer in self.expert_layers],
                "base_expert_prob_layer": tf.keras.layers.serialize(
                    self.base_expert_prob_layer) if self.base_expert_prob_layer else None
                }
