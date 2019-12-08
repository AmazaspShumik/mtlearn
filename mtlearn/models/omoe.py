#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Union

from mmoe import MultiGateMixtureOfExperts
from layers import MixtureOfExpertsLayer

import tensorflow as tf
import numpy as np


class OneGateMixtureOfExperts(MultiGateMixtureOfExperts):
  """
  One-Gate Mixture of Experts - similar to Multi-Gate Mixture of Experts. 
  The only difference is that every task shares the same MOE layer, while in 
  Multi-Gate there is specific MOE layer for each task.
  
  Parameters
  ----------
  experts_layers: list of tensorflow.keras.Layer
    List of layers, where each layer represent corresponding Expert. Note that
    user can build composite layers, so that each layer would represent block of 
    layers or the whole network ( this can be done through subclassing of 
    tensorflow.keras.Layer).

  task_specific_layers: list of tensorflo.keras.Layer
    List of layers, where each layer represents part of the network that 
    specializes in the specific task. Note that user can provide composite layer
    which can be a block of layers or whole neural network (this can be done 
    through subclassing of tensorflow.keras.Layer)

  moe_dropout: bool, optional (Default=False)
    If True, then in the training stage experts are randomly dropped and expert
    competence probabilities are renormalized in the MOE layer.

  moe_dropout_rate: float, optional (default=0.1) 
    Probability that a single expert can be dropped in the MOE layer.

  base_layer: tensorflow.keras.Layer, optional (default=None)
    User defined layer that preprocesses input (for instance splits numeric 
    and categorical features and then normalizes numeric ones and creates 
    embeddings for categorical ones)

  References
  ----------
  [1] Modeling Task Relationships in Multi-task Learning with 
      Multi-gate Mixture-of-Experts, Jiaqi Ma1 et al, 2018 KDD
  """

  def _build_moe_layers(self):
    """Builds Mixture of Experts Layers if they are not provided by the user"""
    self.moe_layer = MixtureOfExpertsLayer(self.expert_layers, 
                                           self.moe_dropout,
                                           self.moe_dropout_rate)
    
  def call(self,
           inputs: Union[np.array, tf.Tensor]
           ) -> List[tf.Tensor]:
    """
    Forward pass of the model.

    Parameters
    ----------
    inputs: np.array or tf.Tensor
      Input to the model

    Returns
    -------
    outputs: list of tf.Tensor
      Outputs of forward pass for each task
    """
    if self.base_layer:
      inputs = self.base_layer(inputs)
    moe = self.moe_layer(inputs)
    outputs = [task_layer(moe) for task_layer in self.task_layers]
    return outputs

