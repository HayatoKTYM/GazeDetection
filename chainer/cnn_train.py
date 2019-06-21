# coding: utf-8
import time

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pandas as pd
from chainer import Chain, Variable
from chainer import cuda
from chainer import optimizers
from sklearn.model_selection import train_test_split


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 50, 5)#50
            self.conv2 = L.Convolution2D(None, 50, 5)
            self.l1 = L.Linear(None, 500)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), 0.2)
        h = self.l2(h)
        return h



