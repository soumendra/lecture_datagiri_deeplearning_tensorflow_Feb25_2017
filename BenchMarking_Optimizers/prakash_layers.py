import tensorflow as tf
import math

def variable_summaries(var):
    """
    calculates the mean, sd, max and min of the variable
    Adds summaries of the variable to tensorboard.
    Adds the histogram of the variable to tensorboard.
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)


def fc(inputs, output_shape, name, activation = True):
    """
    A higher level abstraction to fully connected layer
    input:
    inputs: input to the FC layer
    output_shape: number of neurons in the output layers
    name: name of the operation
    activation: weather to use lrelu as activation function or not. Default is True
    output:
    A FC net
    """
    shape = inputs.shape.as_list()
    with tf.variable_scope(name) as scope:
        weights= tf.get_variable("weights",[shape[1], output_shape] ,tf.float32,  tf.truncated_normal_initializer(stddev=0.1, dtype = tf.float32))
        variable_summaries(weights)
        bias =  tf.get_variable("bias",output_shape,tf.float32, tf.constant_initializer(0.0))
        variable_summaries(bias)
        nets = tf.add(tf.matmul(inputs, weights), bias)
        if activation:
            nets = tf.nn.relu(nets)
    return nets

def model(inputs, outputs):
    nets = fc(inputs,512,"FC_1", activation = True)
    nets = fc(nets, 256, "FC_2", activation = True)
    nets = fc(inputs,512,"FC_3", activation = True)
    nets = fc(nets, outputs.get_shape()[1],"output_layer", activation = False)
    return nets
