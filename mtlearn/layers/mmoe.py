from typing import List, Dict

from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from mtlearn.layers import MixtureOfExpertsLayer
from mtlearn.utils import has_arg


class MultiGateMixtureOfExperts(Layer):
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

    base_layer: tf.keras.layers.Layer, optional (default=None)
      User defined layer that preprocesses input (for instance splits numeric
      and categorical features and then normalizes numeric ones and creates
      embeddings for categorical ones)

    base_expert_prob_layer: tf.keras.layers.Layer, optional (Default=None)
        Layer that extracts features from inputs.

    References
    ----------
    [1] Modeling Task Relationships in Multi-task Learning with
        Multi-gate Mixture-of-Experts, Jiaqi Ma1 et al, 2018 KDD
    """

    def __init__(self,
                 expert_layers: List[Layer],
                 task_layers: List[Layer],
                 moe_layers: List[Layer] = None,
                 moe_dropout: bool = False,
                 moe_dropout_rate: float = 0.1,
                 base_layer: Layer = None,
                 base_expert_prob_layer: Layer = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_layer = base_layer
        self.task_layers = task_layers
        self.n_tasks = len(task_layers)
        self.expert_layers = expert_layers
        self.moe_dropout = moe_dropout
        self.moe_dropout_rate = moe_dropout_rate
        self.base_expert_prob_layer = base_expert_prob_layer
        self.moe_layers = moe_layers if moe_layers else self._build_moe_layers()

    def _build_moe_layers(self) -> List[Layer]:
        """Builds Mixture of Experts Layers if they are not provided by the user"""
        moe_layers = []
        for i in range(self.n_tasks):
            moe_layers.append(MixtureOfExpertsLayer(self.expert_layers,
                                                    self.moe_dropout,
                                                    self.moe_dropout_rate,
                                                    self.base_expert_prob_layer))
        return moe_layers

    def call(self, inputs, training):
        """
        Forward pass of the Multi-Gate Mixture of Experts model.

        Parameters
        ----------
        inputs: np.array or tf.Tensor
          Input to the model

        training: bool
          If True runs model in training mode, otherwise in prediction
          mode.

        Returns
        -------
        outputs: list of tf.Tensor
          Outputs of forward pass for each task
        """
        outputs = []
        if self.base_layer:
            if has_arg(self.base_layer, "training"):
                inputs = self.base_layer(inputs, training)
            else:
                inputs = self.base_layer(inputs)
        moes = [moe(inputs, training) for moe in self.moe_layers]
        for task, moe in zip(self.task_layers, moes):
            if has_arg(task, "training"):
                outputs.append(task(moe, training))
            else:
                outputs.append(task(moe))
        return outputs

    def get_config(self) -> Dict:
        """ Get configuration of the Multi-Gate Mixture of Experts """
        base_config = super().get_config()
        return {**base_config,
                "base_layer": layers.serialize(self.base_layer) if self.base_layer else None,
                "task_layers": [layers.serialize(l) for l in self.task_layers],
                "moe_layers": [layers.serialize(l) for l in self.moe_layers],
                "moe_dropout": self.moe_dropout,
                "moe_dropout_rate": self.moe_dropout_rate
                }
