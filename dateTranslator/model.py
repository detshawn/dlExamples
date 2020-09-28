from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Dropout, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from nmt_utils import *


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    dropout = Dropout(rate=0.8)
    densor2 = Dense(1, activation="relu")
    activator = Activation(softmax,
                           name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=1)

    def one_step_attention(a, s_prev):
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        # For grading purposes, please list 'a' first and 's_prev' second, in this order.
        concat = concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = densor1(concat)
        e = dropout(e)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = dotor([alphas, a])

        return context

    # Please note, this is the post attention LSTM cell.
    # For the purposes of passing the automatic grader
    # please do not modify this global variable.  This will be corrected once the automatic grader is also updated.
    post_activation_LSTM_cell = LSTM(n_s, return_state=True)  # post-attention LSTM
    output_layer = Dense(machine_vocab_size, activation=softmax)

    # init x
    X = Input(shape=(Tx, human_vocab_size), name='X')
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    outputs = []

    # Pre Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Loop for attention
    for t in range(Ty):
        context = one_step_attention(a, s)

        # Post LSTM
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model

