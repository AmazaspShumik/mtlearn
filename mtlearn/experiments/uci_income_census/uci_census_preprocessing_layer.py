#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding


class PreprocessingLayer(Layer):
    """
    Layer that preprocesses input features of UCI Income Census (1994), creates
    embeddings for categorical features. Note this layer assumes numeric features
    are already scaled to 0-1.

    Parameters
    ----------
    training_data: pd.DataFrame
      Data frame with training data

    cat_features_dim: dict
      Dictionary with keys being names of categorical features and value
      being number of categories

    embed_dim: dict
      Dictionary with keys being names of categorical features and value
      being dimenionality of the embedding.

    default_embed_size: int
      If a feature name is not in 'embed_dim' and is conidered sparse then
      dimenioanlity of its embedding will be equal to default_embed_size.

    feature_sparsity_threshold: int
      If dimensionality of categorical feature is below threhold, then it is not
      considered sparse and embedding layer is used instead of one-hot.
    """

    def __init__(self,
                 all_columns: str,
                 cat_features_dim: Dict[str, int],
                 embed_dim: Dict[str, int] = None,
                 feature_sparsity_threshold: int = 3,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.embeddings = {}
        self.one_hot = {}
        self.all_columns = all_columns
        self.cat_columns = list(cat_features_dim.keys())
        self.cat_features_dim = cat_features_dim
        self.embed_dim = embed_dim
        self.feature_sparsity_threshold = feature_sparsity_threshold
        for col in self.cat_columns:
            self._preprocess_cat_feature(col)

    def _preprocess_cat_feature(self,
                                col: str
                                ) -> None:
        """
        Preprocess categorical column: use either embeddings or one-hot. Categorical
        feature can be represented either as an embedding or as one-hot encoding,
        in section 6.3 of paper authors mention that they used embedding for sparse
        features only, unfortunately there is no mention of how they define what is
        sparse feature.

        Parameters
        ----------
        col: str
          Name of the feature
        """
        n_categories = self.cat_features_dim[col]
        if n_categories >= self.feature_sparsity_threshold:
            # build embedding if feature is sparse
            if self.embed_dim and col in self.embed_dim:
                embed_size = self.embed_dim[col]
            else:
                # we follow default setups in fast.ai library see following link:
                # https://github.com/fastai/fastai/blob/master/fastai/tabular/data.py#L13
                embed_size = min(600, round(1.6 * n_categories ** 0.56))
                self.embeddings[col] = Embedding(input_dim=n_categories,
                                                 output_dim=embed_size)
        else:
            # need to build one-hot encoding if feature is not sparse
            self.one_hot[col] = n_categories

    def call(self, inputs):
        """
        Defines forward pass of the Preprocessing Layer,
        Builds embeddings for sparse categorical features and one-hot encodings
        for categorical features with small number of categories. Performs min-max
        scaling for numeric features.

        Parameters
        ----------
        inputs: np.array or tf.Tensor
          Features of UCI Census Income Dataset

        Returns
        -------
        : tf.Tensor
          Output of the preprocessing layer
        """
        all_categorical, numeric_feature_indices = [], []
        for col_index, col in enumerate(self.all_columns):
            if col in self.cat_columns:
                cat_feature = tf.gather(inputs, [col_index], axis=1)
                if col in self.embeddings:
                    # sparse feature -> use embedding
                    cat_representation = tf.gather(self.embeddings[col](cat_feature), 0, axis=-1)
                elif col in self.one_hot:
                    # not sparse -> one-hot encoding
                    if self.one_hot[col] > 2:
                        one_hot_encoded = tf.one_hot(tf.cast(cat_feature, tf.int32), self.one_hot[col])
                        cat_representation = tf.gather(one_hot_encoded, 0, axis=1)
                    else:
                        # if feature is binary, then no need to use use two columns
                        # in one-hot encoding.
                        cat_representation = tf.cast(cat_feature, dtype=tf.float32)
                all_categorical.append(cat_representation)
            else:
                numeric_feature_indices.append(col_index)
        x_numeric = tf.cast(tf.gather(inputs, numeric_feature_indices, axis=1),
                            dtype=tf.float32)
        features = all_categorical + [x_numeric]
        return tf.concat(features, axis=1)

    def get_config(self) -> Dict:
        """ Get configuration for Preprocessing Layer """
        base_config = super().get_config()
        return {**base_config,
                "all_columns": self.all_columns,
                "cat_features_dim": self.cat_features_dim,
                "embed_dim": self.embed_dim,
                "feature_sparsity_threshold": self.feature_sparsity_threshold
                }
