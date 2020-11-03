# coding: utf-8
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = 'Dataset'
D_TYPE = np.float
np.random.seed(0)
# load data, 510-dim feature
X_train = np.array(pd.read_csv(Path(PATH, 'X_train')).iloc[:, 1:], dtype=D_TYPE)
Y_train = np.array(pd.read_csv(Path(PATH, 'Y_train')).iloc[:, 1:], dtype=D_TYPE).flatten()
X_test = np.array(pd.read_csv(Path(PATH, 'X_test')).iloc[:, 1:], dtype=D_TYPE)
# X_train = X_train[:int(len(X_train) * 0.9)]
# normalize, gradient explosion
X_mean = np.mean(X_train, 0).reshape(1, -1)
X_std = np.std(X_train, 0).reshape(1, -1)
X_train = (X_train - X_mean) / (X_std + 1e-8)
train_size = X_train.shape[0]
print(X_train.shape, Y_train.shape, X_test.shape)


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None

    @staticmethod
    def sigmoid(z):
        return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)

    def f(self, X):
        return self.sigmoid(X.dot(self.w) + self.b)

    def predict(self, X):
        return np.round(self.f(X)).astype(np.int)

    @staticmethod
    def cross_entropy_loss(Y_pred, Y_label):
        return np.sum(-Y_label * np.log(Y_pred) - (1 - Y_label) * np.log(1 - Y_pred))

    def gradient(self, X, Y):
        Y_pred = self.f(X)
        error = Y - Y_pred
        w_grad = -np.sum(error * X.T, 1)
        b_grad = -np.sum(error)
        return w_grad, b_grad

    @staticmethod
    def accuracy(Y_pred, Y_label):
        return 1 - np.mean(np.abs(Y_pred - Y_label))

    @staticmethod
    def shuffle(X, Y):
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        return X[randomize], Y[randomize]

    def train(self, X, Y, batch_size=8, epoch=10, learning_rate=0.01):
        self.w = np.zeros((X.shape[1],))
        self.b = np.zeros((1,))

        train_loss = []
        train_acc = []
        n = X.shape[0]
        step = 1
        for _ in range(epoch):
            X, Y = self.shuffle(X, Y)
            for i in range(int(np.floor(n / batch_size))):
                train_x = X[i * batch_size: (i + 1) * batch_size]
                train_y = Y[i * batch_size: (i + 1) * batch_size]
                # gradient descent
                w_grad, b_grad = self.gradient(train_x, train_y)
                self.w = self.w - learning_rate / np.sqrt(step) * w_grad
                self.b = self.b - learning_rate / np.sqrt(step) * b_grad
                step += 1
            y_pred = self.f(X)
            train_loss.append(self.cross_entropy_loss(y_pred, Y) / train_size)
            train_acc.append(self.accuracy(np.round(y_pred), Y))

        plt.subplot(2, 1, 1)
        plt.plot(train_loss)
        plt.title('loss')
        plt.legend(['train'])
        plt.subplot(2, 1, 2)
        plt.plot(train_acc)
        plt.title('acc')
        plt.legend(['train'])
        plt.show()

        print('Training loss: {}'.format(train_loss[-1]))
        print('Training accuracy: {}'.format(train_acc[-1]))


if __name__ == '__main__':
    model = LogisticRegression()
    model.train(X_train, Y_train)
