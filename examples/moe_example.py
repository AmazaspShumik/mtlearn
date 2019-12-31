import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from mtlearn.layers import MixtureOfExpertsLayer

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

N_EXPERTS = 2  # number of experts
LR = 1e-3  # learning rate for the model
VERBOSE = 1
EPOCHS = 500

# build synthetic training data
x = np.asarray(np.expand_dims(np.linspace(-2, 2, 1000), axis=1), dtype=np.float32)
y = 2 * x ** 2 + 1


def get_model():
    """ Creates Mixture of Experts Model """
    # define experts: each expert is simple linear model
    experts = [Dense(1, "linear") for _ in range(N_EXPERTS)]
    # mixture of linear experts
    mole = MixtureOfExpertsLayer(expert_layers=experts, add_dropout=False)
    # build model
    input_layer = Input(shape=(1,))
    output_layer = mole(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mse", optimizer="adam")  # tf.keras.optimizers.Adam(learning_rate=LR))
    return model, experts, mole


# train model
model, experts, mole = get_model()
model.fit(x, y, epochs=EPOCHS, verbose=VERBOSE)

# predict using mixture of experts and for each expert separately
y_test = model.predict(x)
experts_predict = [expert(x).numpy() for expert in experts]

# evaluate model
error = model.evaluate(x, y, batch_size=x.shape[0], verbose=VERBOSE)
print("MSE = {0}".format(error))

# plot aggregated Model / experts  and targets
colors = ["r-", "b-", "g-", "k-", "y-"]
plt.figure(2, figsize=(18, 9))
for expert_pred, color in zip(experts_predict, colors):
    plt.plot(x[:, 0], expert_pred[:, 0], color)
plt.plot(x[:, 0], y_test[:, 0], "co")
plt.plot(x[:, 0], y[:, 0], "mo")
plt.xlabel("input range")
plt.ylabel("Function Value")
plt.title("Mixture of Linear Experts Demo")
plt.show()

# plot probabilities assigned to each expert
expert_probs = mole.expert_probs(x)
plt.figure(3, figsize=(18, 9))
for i in range(len(experts)):
    plt.plot(x[:, 0], expert_probs[:, i], colors[i])
plt.xlabel("input range")
plt.ylabel("Probability")
plt.title("Expert competence probabilities")

plt.show()
