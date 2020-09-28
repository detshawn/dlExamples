import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

def print_hist(hist):
    print_and_save_hist(hist, is_save=False)

def save_hist(hist):
    print_and_save_hist(hist, is_print=False)

def print_and_save_hist(hist, is_print=True, is_save=True):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'yo-', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r+:', label='test loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['accuracy'], 'bo-', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'k+:', label='test acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    if is_save:
        plt.savefig('loss_vs_epochs.png')
    if is_print:
        plt.show()

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
