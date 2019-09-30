import os
import tensorflow as tf
import numpy as np


class generator(object):
    def __init__(self,name):
        self.name = name
        self.reuse = tf.AUTO_REUSE

    def __call__(self,input):



