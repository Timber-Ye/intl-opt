# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:05
# @Author  : Hanchiao
# @File    : multiple_layer_perceptron.py

from . import nnlayers


class get_model:
    def __init__(self, _input_dim, _output_dim, _num_hidden, act_layer=nnlayers.Sigmoid(),
                 output_layer=nnlayers.ActLinear()):
        self.linear1 = nnlayers.Linear(_input_dim, _num_hidden)
        self.linear2 = nnlayers.Linear(_num_hidden, _output_dim)
        self.activation1 = act_layer
        self.output = output_layer

        self.z_2 = None
        self.net_2 = None
        self.z_1 = None
        self.net_1 = None
        self.x = None

    def forward(self, _x):
        self.x = _x
        self.net_1 = self.linear1(self.x)
        self.z_1 = self.activation1(self.net_1)
        self.net_2 = self.linear2(self.z_1)
        self.z_2 = self.output(self.net_2)

        y = self.z_2.reshape(self.z_2.shape[0], -1)

        return y

    def backward(self, _error):
        delta_1 = self.output.backward(self.net_2, _error)
        delta_2 = self.linear2.backward(self.z_1, delta_1)
        delta_3 = self.activation1.backward(self.net_1, delta_2)
        _ = self.linear1.backward(self.x, delta_3)

    def update(self, _lr):
        self.linear1.update(_lr)
        self.linear2.update(_lr)

    def __str__(self):
        return "[Input Layer to Hidden Layer]\n {}\n\n[Hidden Layer to Output Layer]\n{}".format(self.linear1,
                                                                                                 self.linear2)

    def __call__(self, _x):
        return self.forward(_x)

    def state_dict(self):
        state = {
            'Linear_1': self.linear1.state_dict(),
            'Linear_2': self.linear2.state_dict(),
        }
        return state
