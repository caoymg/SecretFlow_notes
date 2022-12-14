# Logistic Regression with jax
# Author: CAO YIMING

import secretflow as sf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import value_and_grad
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

"""
Load the Dataset
"""
def load_train_set(party_id=None, y_flg=None):
    features, label = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, _, y_train, _ = train_test_split(features, label, test_size=0.8, random_state=42)
    
    if party_id != None:
        if party_id == 0:
            if y_flg == None:
                return X_train[:, :15]
            else:
                return y_train
        if party_id == 1:
            if y_flg == None:
                return X_train[:, 15:]
            else:
                return y_train
        else:
            print("invaild party id")
    else:   
        if y_flg == None:
            return X_train
        else:
            return y_train

def load_test_set():
    features, label = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    _, X_test, _, y_test = train_test_split(features, label, test_size=0.8, random_state=42)

    return X_test, y_test

"""
Define the Model
"""
def sigmoid(x):
    return 1/(1 + jnp.exp(-x))

def model(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)

def loss(preds, labels):
    label_probs = preds * labels + (1 - preds) * (1 - labels)
    return -jnp.mean(jnp.log(label_probs))

def train(W, b, x1, x2, y, epochs, learning_rate):
    x = jnp.concatenate([x1, x2], axis=1)
    loss_array = jnp.array([])
    for _ in range(epochs):
        preds = model(W, b, x)
        loss_value, Wb_grad = value_and_grad(loss)(preds, y)
        W -= learning_rate * Wb_grad[0]
        b -= learning_rate * Wb_grad[1]
        loss_array = jnp.append(loss_array, loss_value)
    return loss_array, W, b


"""
Validate the Model
"""
def plot_loss(losses):
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
        
def evaluate(W, b, X_test, y_test):
    y_pred = model(W, b, X_test)
    return roc_auc_score(y_test, y_pred)


"""
Init the Environment
"""
"""In case you have a running secretflow runtime already. """
sf.shutdown()

sf.init(['alice', 'bob'], num_cpus=8, log_to_driver=True)

alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))


# Load the data
x1 = alice(load_train_set)(party_id=0)
x2 = bob(load_train_set)(party_id=1)
y = bob(load_train_set)(party_id=1, y_flg=1)

"""Before trainning, we need to pass hyperparamters and all data to SPU device."""
device = spu
W = jnp.zeros((30))
b = 0.0

W_, b_, x1_, x2_, y_ = (
    sf.to(device, W),
    sf.to(device, b),
    x1.to(device),
    x2.to(device),
    y.to(device),
)

# Train the model
losses, W_, b_ = device(train, static_argnames=['epochs'], num_returns=3)(W_, b_, x1_, x2_, y_, epochs=10, learning_rate=1e-2)

# Plot the loss
"""In order to check losses and model, we need to convert SPUObject(secret) to Python object(plaintest). SecretFlow provide sf.reveal to convert any DeviceObject to Python object.
Be care with sf.reveal???since it may result in secret leak"""
revealed_loss = sf.reveal(losses)
plot_loss(revealed_loss)

# Validate the model
# X_test, y_test = load_test_dataset()
# auc=validate_model(W,b, X_test, y_test)
# print(f'auc={auc}')
X_test, y_test = load_test_set()
auc = evaluate(sf.reveal(W_), sf.reveal(b_), X_test, y_test)
print(f'auc={auc}')
