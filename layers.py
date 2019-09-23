# coding=utf-8
import tensorflow as tf
import numpy as np
from functools import reduce


def get_variable(name, shape, initializer, scope):
    try:
        v = tf.get_variable(name, shape=shape, initializer=initializer)
    except ValueError:
        scope.reuse_variables()
        v = tf.get_variable(name, shape=shape, initializer=initializer)
    return v


def conv2d(input, output_shape, strides=1, kernel_size=3, padding='SAME', name=None, bias=False):
    input_shape = input.get_shape().as_list()[-1]
    filter_shape = [kernel_size, kernel_size, input_shape, output_shape]
    strides_shape = [1, strides, strides, 1]
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=filter_shape, initializer=initializer)
        conv = tf.nn.conv2d(input, filter=weight, strides=strides_shape, padding=padding)
        if (bias):
            _bias = tf.get_variable('bias', shape = filter_shape[-1], initializer = tf.constant_initializer(0.))
            conv = tf.nn.bias_add(conv, _bias)
    return conv



def deconv2d(input, output_channel, output_size, strides=1, kernel_size=3,padding='SAME', name=None, bias=False):
    input_channel = input.get_shape().as_list()[-1]
    batch_shape = input.get_shape().as_list()[0]
    filter_shape = [kernel_size, kernel_size, output_channel, input_channel]
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
    strides_shape = [1, strides, strides, 1]
    output_shape = [batch_shape, output_size, output_size, output_channel]

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape = filter_shape, initializer = initializer)
        conv = tf.nn.conv2d_transpose(input, filter=weight, output_shape=output_shape, strides=strides_shape,
                                      padding=padding)
        if(bias):
            _bias = tf.get_variable('bias', shape = filter_shape[-1], initializer = tf.constant_initializer(0.))
            conv = tf.nn.bias_add(conv, _bias)

    return conv


def dilated_conv2d(input, output_shape, rate=3, kernel_size=3, padding='SAME',name=None, bias=False):
    input_shape = input.get_shape().as_list()[-1]
    filter_shape = [kernel_size, kernel_size, input_shape, output_shape]
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape = filter_shape, initializer = initializer)
        conv = tf.nn.atrous_conv2d(input, filter=weight, rate=rate, padding=padding)
        if (bias):
            _bias = tf.get_variable('bias', shape = filter_shape[-1], initializer = tf.constant_initializer(0.))
            conv = tf.nn.bias_add(conv, _bias)

    return conv


def fc(input, output_shape, name=None, bias=False,initializer = tf.random_normal_initializer(mean=0., stddev=0.02)):
    input_shape = input.get_shape().as_list()
    input = tf.reshape(input, [-1, reduce(lambda x, y: x * y, input_shape[1:])])

    fc_weight_shape = [reduce(lambda x, y: x * y, input_shape[1:]), output_shape]
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape = fc_weight_shape, initializer = initializer)
        fc = tf.matmul(input, weight)
        # fc = tf.layers.dense(input,weight,)
        if (bias):
            _bias = tf.get_variable('bias', shape = fc_weight_shape[-1], initializer = tf.constant_initializer(0.))
            fc = tf.nn.bias_add(fc, _bias)

    return fc


def max_pool(input, ksize, strides_size=1, padding='SAME', name=None):
    with tf.variable_scope(name):
        strides = [1, strides_size, strides_size, 1]
        filters = [1, ksize, ksize, 1]
        max_pool = tf.nn.max_pool(input, filters, strides=strides, padding=padding)

    return max_pool


def avg_pool(input, ksize, strides_size=1, padding='SAME', name=None):
    with tf.variable_scope(name):
        strides = [1, strides_size, strides_size, 1]
        filters = [1, ksize, ksize, 1]
        arg_pool = tf.nn.arg_pool(input, filters, strides=strides, padding=padding)

    return arg_pool


def leaky_relu(input):
    return tf.nn.leaky_relu(input)


def l1_loss(x1, x2):
    return tf.reduce_mean(tf.abs(x1 - x2))


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


def xavier_init(random_size, name, init_type='normal', trainable=True):
    if (init_type == 'zeros'):
        initializer = tf.zeros(shape=random_size)
    else:
        mean = 0
        stddev = 1 / tf.sqrt(random_size[0] / 2.0)
        initializer = tf.random_normal(shape=random_size, mean=mean, stddev=stddev)
    return tf.Variable(initializer, name=name, trainable=trainable)



def bn_layer(x, is_training=True, name=None, moving_decay=0.9, eps=1e-5):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2, 4]

    param_shape = shape[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta

        # 计算当前整个batch的均值与方差
        axes = list(range(len(shape) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training, True), mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        gamma = tf.get_variable('gamma', shape = param_shape, initializer = tf.constant_initializer(1))
        beta = tf.get_variable('beta', shape = param_shape, initializer = tf.constant_initializer(0))

        # 最后执行batch normalization
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
