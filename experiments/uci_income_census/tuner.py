import kerastuner as kt
from sklearn.metrics import roc_auc_score

class IncomeCensusTuner(kt.Tuner):
    """
    Hyperparameter tuner for Income Census dataset
    """

    def run_trial(self,
                  trial,
                  train_features,
                  train_labels_main_task,
                  train_labels_aux_task,
                  val_features,
                  val_labels_main_task
                  ):
        """
        Evaluates set of hyperparameters
        """

        # build model from hyperparameters and train it
        model = self.hypermodel.build(trial.hyperparameters)
        train_labels = (train_labels_main_task, train_labels_aux_task)
        model.fit(x=train_features, y=train_labels, epochs=5, batch_size=1024)

        # predict on validation set
        preds_main_task, preds_aux_task = model.predict(val_features)

        # we follow paper and use only AUC on main task as decribed in section 6.3
        auc = roc_auc_score(val_labels_main_task, preds_main_task)

        # log the score and save model
        self.oracle.update_trial(trial.trial_id, {'auc': auc})
        self.save_model(trial.trial_id, model)







