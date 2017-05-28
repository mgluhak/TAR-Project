import abc
import numpy as np


class ProgressBar:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.max_value = None
        self.step = None
        self.counter = None
        self.initialized = False
        self.print_step = None

    def initialize(self, max_value, start=0, step=1, print_step=1):
        self.max_value = max_value
        self.step = step
        self.counter = start
        self.initialized = True
        self.print_step = print_step

    def increase_counter(self):
        if not self.initialized:
            raise ValueError("Progress bar is not initialized!")
        self.counter += self.step
        if self.step % self.print_step == 0:
            self.show_update()

    @abc.abstractclassmethod
    def show_update(self):
        raise NotImplementedError("Please Implement this method")