#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from typing import List, Union

from ..layers import MixtureOfExpertsLayer

import tensorflow as tf
import numpy as np
from tf.keras.layers import Layer
from tf.keras.models import Model


class MultiGateMixtureOfExperts(Model):
  """
  Multi-Gate Mixture of Experts - multitask learning model that allows 
  automatically learn measure of relationship between tasks and adapts so that
  more relevant are able to share more information. Every task tower starts from 
  its own Mixture of Expert layer that weights competency of each expert for a 
  given task. Task specific layers are built on top of moe layers.
  
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
    competence probabilities are renormalized in each of the MOE layers.

  moe_dropout_rate: float, optional (default=0.1) 
    Probability that a single expert can be dropped in each of MOE layers.

  base_layer: tensorflow.keras.Layer, optional (default=None)
    User defined layer that preprocesses input (for instance splits numeric 
    and categorical features and then normalizes numeric ones and creates 
    embeddings for categorical ones)

  References
  ----------
  [1] Modeling Task Relationships in Multi-task Learning with 
      Multi-gate Mixture-of-Experts, Jiaqi Ma1 et al, 2018 KDD
  """
  def __init__(self,
               expert_layers: List[Layer],
               task_specific_layers: List[Layer],
               moe_dropout: bool=False,
               moe_dropout_rate: float=0.1,
               base_layer: Layer=None,
               **kwargs
              ) -> None:
    super().__init__(**kwargs)
    self.n_tasks = len(task_specific_layers)
    self.base_layer = base_layer
    self.task_layers = task_specific_layers
    self.expert_layers = expert_layers
    self.moe_dropout = moe_dropout
    self.moe_dropout_rate = moe_dropout_rate
    self._build_moe_layers()

  def _build_moe_layers(self):
    """Builds Mixture of Experts Layers if they are not provided by the user"""
    self.moe_layers = []
    for i in range(self.n_tasks):
      self.moe_layers.append( MixtureOfExpertsLayer(self.expert_layers, 
                                                    self.moe_dropout,
                                                    self.moe_dropout_rate))
      
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
    moe = [moe(inputs) for moe in self.moe_layers]
    outputs = [task_layer(moe[i]) for i, task_layer in enumerate(self.task_layers)]
    return outputs
