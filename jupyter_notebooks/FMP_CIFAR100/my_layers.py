import tensorflow as tf
import math

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

X = tf.placeholder(tf.float32, [None, 94, 94, 3])
def conv2d(inputs, filters, kernel_size, strides, name, padding = "SAME"):
    shape = inputs.get_shape().as_list()
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, shape[3], filters], stddev=0.1))
            variable_summaries(weights)
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros(shape = filters,dtype=tf.float32))
            variable_summaries(bias)
        nets = tf.nn.conv2d(inputs, weights, strides = [1, strides, strides, 1] , padding= padding)
        nets = tf.nn.bias_add(nets, bias)
        with tf.name_scope("leaky_relu"):
            nets = tf.maximum(nets, 0.1 * nets)
            variable_summaries(nets)
    return nets

def fmp(inputs, pr = math.sqrt(2.), name = "pool", pseudo_random = False, overlapping = False):
    with tf.name_scope(name):
        nets  = tf.nn.fractional_max_pool(inputs, pooling_ratio = [1, pr, pr,1], pseudo_random = False, overlapping = False)
    return nets[0]

def fc(inputs, output_shape, name):
    shape = inputs.get_shape().as_list()
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            weights= tf.Variable(tf.truncated_normal([shape[1], output_shape], stddev=0.1))
            variable_summaries(weights)
        with tf.name_scope("bias"):
            bias =  tf.Variable(tf.zeros(shape = output_shape,dtype=tf.float32))
            variable_summaries(bias)
        nets = tf.add(tf.matmul(inputs, weights), bias)
    return nets


def cifar_fmp(inputs, pr = math.pow(2, 1/3.0), pseudo_random = False, overlapping= False):
    for i in range(12):
        if i == 0:
            nets = conv2d(inputs, 64, 2, 1, "conv_"+str(i+1))
            nets = fmp(nets, pr, "pool_"+str(i+1), pseudo_random, overlapping)
        else:
            nets = conv2d(nets, 64*(i+1), 2, 1, "conv_"+str(i+1))
            nets = fmp(nets, pr, "pool_" +str(i+1), pseudo_random, overlapping)
    nets = conv2d(nets, 64*12,2,1,"conv13")
    nets = conv2d(nets, 64*12, 1,1, "conv14")
    poolShape = nets.get_shape().as_list()
    nets = tf.reshape(nets, [-1, poolShape[1] * poolShape[2] * poolShape[3]])
    nets = fc(nets, 100, "fc_final")
    return nets
