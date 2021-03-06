import kerastuner as kt
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback
from typing import List


class UCICensusIncomeTuner(kt.Tuner):
    """
    Hyperparameter tuner for Multi-task models on Income Census dataset
    """

    def run_trial(self,
                  trial,
                  train_features,
                  train_labels_main_task,
                  train_labels_aux_task,
                  val_features,
                  val_labels_main_task,
                  val_labels_aux_task,
                  min_batch_size: int = 128,
                  max_batch_size: int = 2048,
                  min_learning_rate: float = 1e-4,
                  max_learning_rate: float = 1e-2,
                  epochs: int = 100,
                  restricted_hyperparameter_search: bool = True,
                  verbose: int = 0,
                  user_callbacks: List[Callback] = None
                  ):
        """
        Evaluates set of hyperparameters

        Parameters
        ----------
        trial: kerastuner.Trial
            Object that holds information about single evaluation
            of set of hyperparameters

        train_features: np.array
            Features for training data

        train_labels_main_task: np.array
            Labels for the main task on training data
            (see section 6.3 of the paper)

        train_labels_aux_task: np.array
            Labels for the auxilary task on training data
            (see section 6.3 of the paper)

        val_features: np.array
            Features for validation data

        val_labels_main_task: np.array
            Labels for the main task on training data
            (see section 6.3 of paper)

        val_labels_aux_task: np.array
            Labels for auxilary task

        min_batch_size: int
            Minimum possible size of the batch

        max_batch_size: int
            Maximum possible size of the batch

        min_learning_rate: float
            Minimum possible value of the learning rate

        max_learning_rate: float
            Maximum possible value of the learning rate

        epochs: int
            Number of epochs

        restricted_hyperparameter_search: bool
            In section 4.2 of the [1] paper it is mentioned that for MMOE
            they used only Dense layers with ReLU activation units, in
            section 6.4 of the same paper it is mentioned that ReLU is used
            in shared bottom and single task architectures. In the same section
            it is mentioned that for SGD and batch_size=1024 was used for
            optimization of all model. So if restricted_hyperparameter_search
            is set to True, then
             - batch_size = 1024
             - hidden_layer_activation = relu
             - optimizer = sgd

        verbose: int
            If 1 print intermediate values of metrics and loss values, 0
            do not print

        user_callbacks: List of callbacks
            User defined callbacks

        References
        ----------
        [1] Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
            by Jiaqi Ma et al.
        """
        hp = trial.hyperparameters
        if not restricted_hyperparameter_search:
            batch_size = hp.Int("batch_size",
                                min_value=min_batch_size,
                                max_value=max_batch_size)
            optimizer_name = hp.Choice("optimizer", ["sgd", "adam", "rms"])
        else:
            batch_size = hp.Fixed("batch_size", 1024)
            optimizer_name = hp.Fixed("optimizer", "sgd")

        # define learning rate and optimizer
        lr = hp.Float("learning_rate",
                      min_learning_rate,
                      max_learning_rate,
                      sampling="linear")
        if optimizer_name == "adam":  # probably could use enums here
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == "sgd":
            optimizer = SGD(learning_rate=lr)
        elif optimizer_name == "rms":
            optimizer = RMSprop(learning_rate=lr)
        else:
            raise ValueError("Unknown optimizer_name {0}".format(optimizer_name))

        # build model from hyperparameters
        model = self.hypermodel.build(hp)

        # compile model
        val_data = (val_features, [val_labels_main_task, val_labels_aux_task])
        model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                      optimizer=optimizer,
                      metrics=[tf.keras.metrics.AUC()]
                      )

        # train
        train_labels = (train_labels_main_task, train_labels_aux_task)
        model.fit(x=train_features,
                  y=train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=verbose,
                  validation_data=val_data,
                  callbacks=user_callbacks)

        # predict on validation set
        preds_main_task, preds_aux_task = model.predict(val_features)

        # we follow paper and use only AUC on main task for performance
        # evaluation of hyperparameter optimizer (as decribed in section 6.3.1)
        auc = roc_auc_score(val_labels_main_task, preds_main_task)

        # log the score and save model
        self.oracle.update_trial(trial.trial_id, {'auc': auc})
        self.save_model(trial.trial_id, model)
