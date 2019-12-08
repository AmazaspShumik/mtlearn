#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Union, Dict, List

import tensorflow as tf
import numpy as np
from tf.keras.layers import Layer, Dropout, Dense


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
               dropout_rate: float=0.1,
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
                              noise_shape=(1,*batch_input_shape[1:]),
                              **self.kwargs
                             )
    super().build(batch_input_shape)
    
  def call(self, 
           inputs: Union[np.array, tf.Tensor],
           training: bool,
           ) -> tf.Tensor:
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
    return expert_probs_drop / tf.reduce_sum(expert_probs_drop,
                                             keepdims=True,
                                             axis=1)
    
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

  References
  ----------
  [1] Recommending What Video to Watch Next: A Multitask Ranking System, 2019.
      Zhe Zhao
  [2] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
      Layer, 2017. Noam Shazeer, Azalia Mirhoseini.
  """
  def __init__(self, 
               expert_layers: List[Layer],
               add_dropout: bool=False,
               dropout_rate: float=0.1,
               **kwargs
               ) -> None:
    super().__init__(**kwargs)
    self.kwargs = kwargs
    self.expert_layers = expert_layers
    self.n_experts = len(expert_layers)
    self.add_dropout = add_dropout
    self.dropout_rate = dropout_rate

  def build(self, 
            batch_input_shape
            ) -> None:
    """
    Builds softmax layer that estimates probability of expert utilization 
    in every given region and then builds expert dropout layer with 
    renormalization.

    Parameters
    ----------
    batch_input_shape: tuple
      Shape of the inputs
    """
    self.expert_probs = Dense(self.n_experts, "softmax", **self.kwargs)
    if self.add_dropout:
      self.drop_expert_layer = ExpertUtilizationDropout(self.dropout_rate)
    super().build(batch_input_shape) 
      
  def call(self, 
           inputs: Union[np.array, tf.Tensor],
           training: bool
          ) -> tf.Tensor:
    """
    Defines set of computations performed in the layer

    Parameters
    ----------
    inputs: array-like structure 
      Inputs to the layer (can be numpy array, tf.Tensor, list/tuple of numpy 
      array or tensorflow Tensors)

    training: bool
      True if layer is called in training mode, False otherwise 

    Returns
    -------
    moe_output: tf.Tensor
      Output of mixture of experts layers ( linearly weighted output of expert
      layers).
    """
    # compute each expert output
    experts_output = [expert(inputs) for expert in self.expert_layers]

    # compute probability of expert (degree of expert utilization) for given
    # input set
    expert_utilization_prob = self.expert_probs(inputs)
    if self.add_dropout:
      expert_utilization_prob = self.drop_expert_layer(expert_utilization_prob,
                                                       training)
      
    # compute weighted output of experts 
    moe_output = 0
    for i, expert_output in enumerate(experts_output):
      moe_output += (expert_output
                     *tf.expand_dims(expert_utilization_prob[:,i], axis=-1)
                    )
    return moe_output

  def get_config(self) -> Dict:
    """ Config of the MOE layer """
    base_config = super().get_config()
    return {**base_config,
            "add_dropout": self.add_dropout,
            "dropout_rate": self.dropout_rate,
            "expert_layers": [tf.keras.layers.serialize(layer) for 
                              layer in self.expert_layers]
            }