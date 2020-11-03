import tensorflow as tf

import numpy as np
from functools import reduce
import time


@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)

class AlexNet:
    """
    A trainable version alexnet.
    """

    def __init__(self, x, trainable=True, dropout=0.1):
        self.G = tf.get_default_graph()
        self.dropout = dropout
        self.rgb = x
        self.is_training = trainable
        self.build()

    def build(self, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        start_time = time.time()

        # self.rgb = tf.cond(self.is_training,
        #         lambda: self.imageDisorder(self.rgb),
        #         lambda: self.rgb
        # )

        self.convbin1 = self.conv_layer(self.rgb, filter_width = 11, filter_height = 11, filter_count = 96, stride_x = 4, stride_y = 4, name = "layer1", padding='VALID', binary = False)
        self.lrnbin1 = self.local_response_normalization(self.convbin1)
        self.poolbin1 = self.max_pool(self.lrnbin1, filter_width=3, filter_height=3, stride_x=2, stride_y=2, padding='VALID', name="pool1")

        self.conv1 = self.conv_layer_full(self.rgb, filter_width = 11, filter_height = 11, filter_count = 96, stride_x = 4, stride_y = 4, name = "layer1", padding='VALID', binary = False)
        self.lrn1 = self.local_response_normalization(self.conv1)
        self.pool1 = self.max_pool(self.lrn1, filter_width=3, filter_height=3, stride_x=2, stride_y=2, padding='VALID', name="pool1")
        self.loss1 = tf.reduce_mean(tf.square(self.pool1 - self.poolbin1))


        self.convbin2 = self.conv_layer(self.poolbin1, filter_width = 5, filter_height = 5, filter_count = 256, stride_x = 1, stride_y = 1, name = "layer2", padding='SAME', binary = False)
        self.lrnbin2 = self.local_response_normalization(self.convbin2)
        self.poolbin2 = self.max_pool(self.lrnbin2, filter_width=3, filter_height=3, stride_x=2, stride_y=2, padding='VALID', name="pool2")

        self.conv2 = self.conv_layer_full(self.pool1, filter_width = 5, filter_height = 5, filter_count = 256, stride_x = 1, stride_y = 1, name = "layer2", padding='SAME', binary = False)
        self.lrn2 = self.local_response_normalization(self.conv2)
        self.pool2 = self.max_pool(self.lrn2, filter_width=3, filter_height=3, stride_x=2, stride_y=2, padding='VALID', name="pool2")
        self.loss2 = tf.reduce_mean(tf.square(self.pool2 - self.poolbin2))


        self.convbin3 = self.conv_layer(self.poolbin2, filter_width = 3, filter_height = 3, filter_count = 384, stride_x = 1, stride_y = 1, name = "layer3", padding='SAME', binary = True)
        self.conv3 = self.conv_layer_full(self.pool2, filter_width = 3, filter_height = 3, filter_count = 384, stride_x = 1, stride_y = 1, name = "layer3", padding='SAME', binary = False)
        self.loss3 = tf.reduce_mean(tf.square(self.convbin3 - self.conv3))


        self.convbin4 = self.conv_layer(self.convbin3, filter_width = 3, filter_height = 3, filter_count = 384, stride_x = 1, stride_y = 1, name = "layer4", padding='SAME', binary = True)
        self.conv4 = self.conv_layer_full(self.conv3, filter_width = 3, filter_height = 3, filter_count = 384, stride_x = 1, stride_y = 1, name = "layer4", padding='SAME', binary = False)
        self.loss4 = tf.reduce_mean(tf.square(self.convbin4 - self.conv4))


        self.convbin5 = self.conv_layer(self.convbin4, filter_width = 3, filter_height = 3, filter_count = 256, stride_x = 1, stride_y = 1, name = "layer5", padding='SAME', binary = False)
        self.poolbin5 = self.max_pool(self.convbin5, filter_width=3, filter_height=3, stride_x=2, stride_y=2, padding='VALID', name="pool5")

        self.conv5 = self.conv_layer_full(self.conv4, filter_width = 3, filter_height = 3, filter_count = 256, stride_x = 1, stride_y = 1, name = "layer5", padding='SAME', binary = False)
        self.pool5 = self.max_pool(self.conv5, filter_width=3, filter_height=3, stride_x=2, stride_y=2, padding='VALID', name="pool5")
        self.loss5 = tf.reduce_mean(tf.square(self.pool5 - self.poolbin5))

        pool5_shape = self.poolbin5.get_shape().as_list()
        flattened_input_size = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        self.fcbin6 = self.fc_layer(self.poolbin5, flattened_input_size, 4096, "layer6")
        self.relubin6 = tf.nn.relu(self.fcbin6)
        self.relubin6 = tf.cond(self.is_training,
                lambda: tf.nn.dropout(self.relubin6, self.dropout),
                lambda: self.relubin6
        )

        self.fc6 = self.fc_layer_full(self.pool5, flattened_input_size, 4096, "layer6")
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.cond(self.is_training,
                lambda: tf.nn.dropout(self.relu6, self.dropout),
                lambda: self.relu6
        )
        self.loss6 = tf.reduce_mean(tf.square(self.relu6 - self.relubin6))


        self.fcbin7 = self.fc_layer(self.relubin6, 4096, 4096, "layer7")
        self.relubin7 = tf.nn.relu(self.fcbin7)

        self.relubin7 = tf.cond(self.is_training,
                lambda: tf.nn.dropout(self.relubin7, self.dropout),
                lambda: self.relubin7
        )

        self.fc7 = self.fc_layer_full(self.relu6, 4096, 4096, "layer7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.relu7 = tf.cond(self.is_training,
                lambda: tf.nn.dropout(self.relu7, self.dropout),
                lambda: self.relu7
        )
        self.loss7 = tf.reduce_mean(tf.square(self.relu7 - self.relubin7))


        self.fcbin8 = self.fc_layer(self.relubin7, 4096, 100, "layer8")
        self.fc8 = self.fc_layer_full(self.relu7, 4096, 100, "layer8")
        # self.prob = tf.nn.softmax(self.fc8, name="prob")
        self.loss8 = tf.reduce_mean(tf.square(self.fc8 - self.fcbin8))

        print(("build model finished: %ds" % (time.time() - start_time)))



    def max_pool(self, bottom,filter_width, filter_height, stride_x, stride_y, name, padding='VALID'):
        return tf.nn.max_pool(
            bottom, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name
        )


    def conv_layer(self, input, filter_width, filter_height, filter_count, stride_x, stride_y, name, padding='VALID', binary = False):
        input_channels = input.get_shape()[-1].value
        with tf.variable_scope(name):
            filt, biases = self.get_conv_var(filter_height, filter_width, input_channels,filter_count,name)
            # filt, biases = self.get_conv_var(filter_height, filter_width, input_channels,filter_count,'conv')

            if binary == True:
                print(name, "is binary layer")
                filt = self.quantize(filt)
                biases = self.quantize(biases)
            conv = tf.nn.conv2d(input, filt, [1, stride_y, stride_x, 1], padding=padding)
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)
            return relu


    def local_response_normalization(self, input, name='lrn'):
        # From article: Local Response Normalization: we used k=2, n=5, α=10^−4, and β=0.75.
        with tf.name_scope(name):
            lrn = tf.nn.local_response_normalization(input=input, depth_radius=2, alpha=10 ** -4,
            beta=0.75, name='local_response_normalization')
            return lrn

    def fc_layer(self, bottom, in_size, out_size, name, binary = False):
        with tf.variable_scope(name):
            # weights, biases = self.get_fc_var(in_size, out_size, 'fc')
            weights, biases = self.get_fc_var(in_size, out_size, name)

            if binary == True:
                print(name, "is binary layer")
                weights = self.quantize(weights)
                biases = self.quantize(biases)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.matmul(x, weights)
            bias = tf.nn.bias_add(fc, biases)
            return fc

    def get_conv_var(self, filter_height, filter_width, input_channels,filter_count,name):
        filterShape = [filter_height, filter_width, input_channels, filter_count]
        biasShape = [filter_count]

        initial_value = tf.truncated_normal(
            [filter_height, filter_width, input_channels, filter_count], 0.0, 0.001
        )
        print("load: ", name , "_conv_filters.txt")
        initial_value =  tf.constant(np.loadtxt('bin-weights/' + name + '_' +  name + '_filters.txt', dtype=np.float32).reshape(filterShape))

        filters = self.get_var(initial_value, name + "_filters")

        initial_value = tf.truncated_normal(
            [filter_count], 0.0, 0.001
        )

        # initial_value = tf.constant(np.loadtxt('bin-weights/' + name + '_conv_biases.txt', dtype=np.float32).reshape(biasShape))
        initial_value = tf.constant(np.loadtxt('bin-weights/' + name + '_' + name + '_biases.txt', dtype=np.float32).reshape(biasShape))

        biases = self.get_var(initial_value, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        weightShape = [in_size, out_size]
        biasShape = [out_size]
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        initial_value =  tf.constant(np.loadtxt('bin-weights/' + name + '_' + name + '_weights.txt', dtype=np.float32).reshape(weightShape))
        weights = self.get_var(initial_value,  name + "_weights")


        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        initial_value = tf.constant(np.loadtxt('bin-weights/' + name + '_' + name + '_biases.txt', dtype=np.float32).reshape(biasShape))
        biases = self.get_var(initial_value,  name + "_biases")

        return weights, biases

    def get_var(self, initial_value, var_name):
        value = initial_value
        var = tf.Variable(value, name=var_name)
        return var


    ## fc8
    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def imageDisorder(self, x):
        x = tf.image.random_flip_left_right(x)
        # x = tf.image.random_flip_up_down(x)
        x = tf.image.random_hue(x,max_delta = 0.5)
        x = tf.image.random_saturation(x,lower=0, upper=2.0)
        x = tf.image.random_brightness(x, max_delta=63)
        x = tf.image.random_contrast(x,lower=0.1, upper=2.0)
        return x

    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)

    ##############################################################################
    #       以下是同时双向网络中全精度网络计算所用到的函数
    #       虽然不简洁，但这种方式比较有效
    ############################################################################
    def conv_layer_full(self, input, filter_width, filter_height, filter_count, stride_x, stride_y, name, padding='VALID', binary = False):
        input_channels = input.get_shape()[-1].value
        with tf.variable_scope(name):
            filt, biases = self.get_constant_conv_var(filter_height, filter_width, input_channels,filter_count,name)
            conv = tf.nn.conv2d(input, filt, [1, stride_y, stride_x, 1], padding=padding)
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)
            return relu


    def fc_layer_full(self, bottom, in_size, out_size, name, binary = False):
        with tf.variable_scope(name):
            # weights, biases = self.get_fc_var(in_size, out_size, 'fc')
            weights, biases = self.get_constant_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.matmul(x, weights)
            bias = tf.nn.bias_add(fc, biases)
            return fc


    def get_constant_conv_var(self, filter_height, filter_width, input_channels,filter_count,name):
        filterShape = [filter_height, filter_width, input_channels, filter_count]
        biasShape = [filter_count]

        filters =  tf.constant(np.loadtxt('bin-weights/' + name + '_' +  name + '_filters.txt', dtype=np.float32).reshape(filterShape))
        biases = tf.constant(np.loadtxt('bin-weights/' + name + '_' + name + '_biases.txt', dtype=np.float32).reshape(biasShape))
        return filters, biases




    def get_constant_fc_var(self, in_size, out_size, name):
        weightShape = [in_size, out_size]
        biasShape = [out_size]
        weights =  tf.constant(np.loadtxt('bin-weights/' + name + '_' + name + '_weights.txt', dtype=np.float32).reshape(weightShape))
        biases = tf.constant(np.loadtxt('bin-weights/' + name + '_' + name + '_biases.txt', dtype=np.float32).reshape(biasShape))
        return weights, biases

