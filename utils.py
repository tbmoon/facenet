import numpy as np
import torch
import visdom
from path import Path


class ModelSaver():

    def __init__(self):
        self.previous_acc = 0.
        self.current_acc = 0.

    def __set_accuracy(self, accuracy):
        self.previous_acc = self.current_acc
        self.current_acc = accuracy

    def save_if_best(self, accuracy, state):
        self.__set_accuracy(accuracy)
        if self.current_acc > self.previous_acc:
            torch.save(state, 'log/best_state.pth')


def create_if_not_exist(path):
    path = Path(path)
    if not path.exists(): path.touch()


def init_log_just_created(path):
    create_if_not_exist(path)
    with open(path, 'r') as f:
        if len(f.readlines()) <= 0:
            init_log_line(path)


def init_log_line(path):
    with open(path, 'w') as f:
        f.write('epoch,acc,loss\n')


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name],
                                 name=split_name)
