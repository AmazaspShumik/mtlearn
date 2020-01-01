from typing import List
from unittest import TestCase
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model

from mtlearn.layers import MixtureOfExpertsLayer
from tests import BaseSpecificationMTL


def get_mtl_model_with_moe(input_shape: int,
                           apply_dropout: bool,
                           dropout_rate: float,
                           n_experts: int = 20,
                           ) -> Model:
    """
    Create Mixture of Linear Model Experts
    Parameters
    ----------
    input_shape: int
        Number of features

    apply_dropout: bool
        Apply dropout

    dropout_rate: float
        Probability with which neurons will be randomly
        omitted during training.

    n_experts: int (default=20)
        Number of experts

    Returns
    -------
    model: tf.keras.models.Model
        Compiled model
    """
    # define experts: each expert is simple linear model
    experts = [Dense(1, "linear") for _ in range(n_experts)]
    # mixture of linear experts
    mole = MixtureOfExpertsLayer(expert_layers=experts,
                                 add_dropout=apply_dropout,
                                 dropout_rate=dropout_rate)
    # build model
    input_layer = Input(shape=(input_shape,))
    output_layer = mole(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mse", optimizer="adam")
    return model


class TestMOE(TestCase, BaseSpecificationMTL):
    """ Unit tests for Mixture of Experts Layer """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_data_and_directory()

    @property
    def models(self) -> List[Model]:
        """
        Create models with Mixture of Experts Layer, one model
        with dropout and another one without dropout.
        """
        model_with_dropout = get_mtl_model_with_moe(1, True, 0.5)
        model_without_dropout = get_mtl_model_with_moe(1, False, 0.01)
        return [model_with_dropout, model_without_dropout]

    def test_goodness_of_fit(self) -> None:
        """
        Checks Goodness of fit for single task problem
        """
        for model in self.models:
            model.fit(x=self.x, y=self.y_one, epochs=1000, verbose=1)
            preds = model.predict(self.x)
            self.assertTrue(preds.shape[0] == self.y_one.shape[0])
            self.assertTrue(np.mean(np.square(preds[:, 0] - self.y_one[:, 0])) < 1e-1)

    def test_save_load_mtl(self) -> None:
        """
        Tests saving and loading
        """
        # create directory for saving model
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        # load and reload
        for model in self.models:
            preds = model.predict(self.x)
            saved_model_path = os.path.join(self.test_dir, "test_model_saving")
            model.save(saved_model_path)
            reloaded_model = load_model(saved_model_path)
            preds_reloaded = reloaded_model.predict(self.x)

        # delete temporary directory
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

        # check equality of predictions from original and reloaded models
        self.assertTrue(np.all(preds_reloaded == preds))


if __name__=="__main__":
    test_moe = TestMOE()
    test_moe.test_goodness_of_fit()