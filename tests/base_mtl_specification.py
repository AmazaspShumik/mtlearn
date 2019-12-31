import os
import shutil

import numpy as np
from tensorflow.keras.models import load_model


class BaseSpecificationMTL:
    """
    Base class for testing multitask and single tak models
    """

    def set_data_and_directory(self) -> None:
        """ Create test directory and test data """
        # create test directory
        self.test_dir = os.path.join(os.getcwd(), "test_dir_mtl_models")

        # Create simple dataset for testing multitask models, the idea
        # is that ability to overfit indicates that model has capacity
        # to learn.
        self.x = np.expand_dims(np.linspace(-2, 2, 1000), 1)
        self.y_one = 2 * self.x ** 2 + 1
        self.y_two = 2 * self.x ** 2 - 1

    @property
    def models(self):
        raise NotImplementedError()

    def test_save_load_mtl(self) -> None:
        """ Checks saving and loading functionality of multitask model """

        # create directory for saving model
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        # run prediction before saving, reload and predict again
        for model in self.models:
            preds_one, preds_two = model.predict(self.x)
            saved_model_path = os.path.join(self.test_dir, "test_model_saving")
            model.save(saved_model_path)
            reloaded_model = load_model(saved_model_path)
            preds_one_reloaded, preds_two_reloaded = reloaded_model.predict(self.x)

            # delete temporary directory
            if os.path.isdir(self.test_dir):
                shutil.rmtree(self.test_dir)

            # check equality of predictions from original and reloaded models
            self.assertTrue(np.all(preds_one == preds_one_reloaded))
            self.assertTrue(np.all(preds_two == preds_two_reloaded))

    def test_goodness_of_fit(self) -> None:
        """
        Checks how well model fitted training data, we want to check
        that model can perfectly fit data (i.e. overfit)
        """
        for model in self.models:
            model.fit(x=self.x, y=[self.y_one, self.y_two], epochs=500, verbose=1)
            preds_one, preds_two = model.predict(self.x)
            n_samples_one, n_samples_two = preds_one.shape[0], preds_two.shape[0]
            self.assertTrue(n_samples_one == self.y_one.shape[0])
            self.assertTrue(n_samples_two == self.y_two.shape[0])
            self.assertTrue(np.mean(np.square(preds_one[:, 0] - self.y_one[:,0])) < 2e-1)
            self.assertTrue(np.mean(np.square(preds_two[:, 0] - self.y_two[:,0])) < 2e-1)
