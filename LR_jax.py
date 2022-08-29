# Logistic Regression with jax
# Author: CAO YIMING

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
def load_train_set(party_id=None):
    features, label = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, _, y_train, _ = train_test_split(features, label, test_size=0.8, random_state=42)
    
    if party_id != None:
        if party_id == 0:
            return X_train[:, :15], y_train
        if party_id == 1:
            return X_train[:, 15:], y_train
        else:
            return "invaild party id"
    else:   
        return X_train, y_train

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
        grad_fun = value_and_grad(loss)
        loss_value, Wb_grad = grad_fun(preds, y)
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


# Load the data
x1, _ = load_train_set(party_id=1)
x2, y = load_train_set(party_id=0)

# Hyperparameter
W = jnp.zeros((30))
b = 0.0
epochs = 10
learning_rate = 1e-2

# Train the model
losses, W, b = train(W, b, x1, x2, y, epochs=10, learning_rate=1e-2)

# Plot the loss
plot_loss(losses)

# Validate the model
X_test, y_test = load_test_set()
auc=evaluate(W,b, X_test, y_test)
print(f'auc={auc}')
