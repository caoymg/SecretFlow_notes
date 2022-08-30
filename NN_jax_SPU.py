# Neural Network with jax
# Author: CAO YIMING

import secretflow as sf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from jax.example_libraries import stax
from jax.example_libraries.stax import (
    Dense,
    Relu,
)
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers, stax
from sklearn.metrics import roc_auc_score

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
def MLP():
    nn_init, nn_apply = stax.serial(
        Dense(30),
        Relu,
        Dense(15),
        Relu,
        Dense(8),
        Relu,
        Dense(1),
    )

    return nn_init, nn_apply


KEY = jax.random.PRNGKey(11)
INPUT_SHAPE = (-1,30)

def init_state(learning_rate):
    init_fun, _ = MLP()
    # params_init: the MLP model parameters
    _, params_init = init_fun(KEY, INPUT_SHAPE)
    opt_init, _, _ = optimizers.sgd(learning_rate)
    opt_state = opt_init(params_init)
    return opt_state

def mse(y, pred):
    return jnp.mean(jnp.multiply(y - pred, y - pred) / 2.0)
    
def update_model(model_fun, opt_get_params, opt_update, state, imgs, labels, i):

    def loss(params):
        y = model_fun(params, imgs)
        return mse(y, labels), y

    grad_fn = jax.value_and_grad(loss, has_aux=True)
    (loss, y), grads = grad_fn(opt_get_params(state))
    return opt_update(i, grads, state)
    
def train( 
    x1, x2, y,
    opt_state,
    learning_rate,
    epochs,
    batch_size,
):
    x = jnp.concatenate([x1, x2], axis=1)

    _, model_fun = MLP()
    _, opt_update, opt_get_params = optimizers.sgd(learning_rate)

    for i in range(1, epochs + 1):
        x_batchs = jnp.array_split(x, len(x) / batch_size, axis=0)
        y_batchs = jnp.array_split(y, len(y) / batch_size, axis=0)

        for _, (batch_x, batch_y) in enumerate(zip(x_batchs, y_batchs)):
            opt_state = update_model(model_fun, opt_get_params, opt_update, opt_state, batch_x, batch_y, i)
    return opt_get_params(opt_state)


"""
Validate the Model
"""
def evaluate(params, X_test, y_test):
    _, model_fun = MLP()
    y_pred = model_fun(params, X_test)
    return roc_auc_score(y_test, y_pred)


sf.shutdown()

sf.init(['alice', 'bob'], num_cpus=8, log_to_driver=True)

alice, bob = sf.PYU('alice'), sf.PYU('bob')
# secure processing unit
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Load the data
x1 = alice(load_train_set)(party_id=0)
x2 = bob(load_train_set)(party_id=1)
y = bob(load_train_set)(party_id=1, y_flg=1)

batch_size = 5
epochs = 15
learning_rate = 0.1

"""Before trainning, we need to pass hyperparamters and all data to SPU device."""
device = spu
x1_, x2_, y_ = x1.to(device), x2.to(device), y.to(device)
init_params_ = sf.to(spu, lambda: init_state(learning_rate))


params_spu = spu(train, static_argnames=['learning_rate', 'epochs', 'batch_size'])(
    x1_, x2_, y_,init_params_, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size
)

params = sf.reveal(params_spu)
print(params)

X_test, y_test = load_test_set()
auc = evaluate(params, X_test, y_test)
print(f'auc={auc}')

