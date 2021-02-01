from constants import gridAddress, region_name

import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from datetime import date
import boto3

import torch as th
from torch import nn

import os
from websocket import create_connection
import websockets
import json
import requests
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()
import binascii

sy.make_hook(globals())
hook.local_worker.framework = None  # force protobuf serialization for tensors
th.random.manual_seed(1)


# Define some standard loss functions
def mse_with_logits(logits, targets, batch_size):
    """ Calculates mse
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    return (logits - targets).sum() / batch_size


def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs).sum() / batch_size


def absolute_error(logits, targets, batch_size):
    """ Calculates absolute error
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    return abs(logits - targets) / batch_size


def binary_cross_entropy(logits, targets, batch_size):
    """ Calculates binary cross entropy lose
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    sum_score = (logits * targets.sum(dim=1, keepdim=True).log()).sum()
    mean_sum_score = sum_score / batch_size
    return -mean_sum_score


# Define standard optimizers
def naive_sgd(param, **kwargs):
    """ Naive Standard Gradient Descent"""
    return param - kwargs['lr'] * param.grad


# Standard function will set tensors as model parameters
def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx


def def_training_plan(model, X, y, plan_dict=None):
    """
    :param model: A model built in pytorch
    :param X: Input data
    :param y: Labels
    :param plan_dict: A dictionary representing attributes of the training plan. Values are set to defaults if not set.
    :return: Model parameters and a training plan to be used with pysyft functions
    """
    model_pkl = jsonpickle.encode(model_params)
    x_pkl = jsonpickle.encode(X)
    y_pkl = jsonpickle.encode(y)

    if plan_dict is None:
        plan_dict = {}
    if 'loss' in plan_dict:
        loss_func = plan_dict['loss']
    else:
        loss_func = softmax_cross_entropy_with_logits
    loss_func_pkl = jsonpickle.encode(loss_func)

    if 'optimizer' in plan_dict:
        optim_func = plan_dict['optimizer']
    else:
        optim_func = naive_sgd
    optim_func_pkl = jsonpickle.encode(optim_func)

    if 'training_plan' in plan_dict:
        training_plan_pkl = jsonpickle.encode(plan_dict['training_plan'](X, y, batch_size, lr, model_params))
    else:
        training_plan_pkl = None
    model_params = [param.data for param in model.parameters()]

    training_plan = {"model":model_pkl, "x":x_pkl, "y":y_pkl,
                     "training_plan_func":training_plan_pkl, "optim":optim_func_pkl, 'loss':loss_func_pkl}
    return training_plan


# Define standard averaging plan
def def_avg_plan(model_params, func=None):
    if func is not None:
        @sy.func2plan()
        def avg_plan(avg, item, num):
            func(avg, item, num)
    else:
        @sy.func2plan()
        def avg_plan(avg, item, num):
            new_avg = []
            for i, param in enumerate(avg):
                new_avg.append((avg[i] * num + item[i]) / (num + 1))
            return new_avg
    avg_plan_pkl = jsonpickle.encode(avg_plan)
    # Build the Plan

    return avg_plan_pkl


def send_model(name, version, batch_size, learning_rate, max_updates, model_params, grid, training_plan, avg_plan):
    """ Function to send model to node """


class LinearRegression(th.nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = th.nn.Linear(3, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


if __name__ == "__main__":
    model = LinearRegression()
    X = th.randn(1, 3)
    y = nn.functional.one_hot(th.tensor([2]))
    training_plan = def_training_plan(model, X, y, {"loss": mse_with_logits})
    resp = requests.post('http://127.0.0.1:5000/send', json=training_plan)