
#coding=utf-8

import tensorflow as tf

import numpy as np
from functools import reduce
import time

VGG_MEAN = [103.939, 116.779, 123.68]
BN_EPSILON = 1e-4

@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, x, trainable=True, dropout=0.1):
        # if vgg19_npy_path is not None:
        #     self.data_dict = np.load(vgg19_npy_path, encoding="latin1").item()
        # else:
        #     self.data_dict = None

        # print("============ data_dict:", data_dict)
        self.G = tf.get_default_graph()
        self.var_dict = {}
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

        self.rgb = tf.cond(self.is_training,
                lambda: self.imageDisorder(self.rgb),
                lambda: self.rgb
        )
        self.convbin1_1 = self.conv_layer(self.rgb, 3, 64, "conv1_1", True)
        self.convbin1_2 = self.conv_layer(self.convbin1_1, 64, 64, "conv1_2", True)
        self.poolbin1 = self.max_pool(self.convbin1_2, "pool1")

        self.conv1_1 = self.conv_layer_full(self.rgb, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer_full(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, "pool1")
        self.loss1 = tf.reduce_mean(tf.square(self.pool1 - self.poolbin1))


        self.convbin2_1 = self.conv_layer(self.poolbin1, 64, 128, "conv2_1", True)
        self.convbin2_2 = self.conv_layer(self.convbin2_1, 128, 128, "conv2_2", True)
        self.poolbin2 = self.max_pool(self.convbin2_2, "pool2")

        self.conv2_1 = self.conv_layer_full(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer_full(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, "pool2")
        self.loss2 = tf.reduce_mean(tf.square(self.pool2 - self.poolbin2))


        self.convbin3_1 = self.conv_layer(self.poolbin2, 128, 256, "conv3_1", True)
        self.convbin3_2 = self.conv_layer(self.convbin3_1, 256, 256, "conv3_2", True)
        self.convbin3_3 = self.conv_layer(self.convbin3_2, 256, 256, "conv3_3", True)
        self.convbin3_4 = self.conv_layer(self.convbin3_3, 256, 256, "conv3_4", True)
        self.poolbin3 = self.max_pool(self.convbin3_4, "pool3")

        self.conv3_1 = self.conv_layer_full(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer_full(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer_full(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer_full(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, "pool3")
        self.loss3 = tf.reduce_mean(tf.square(self.pool3 - self.poolbin3))


        self.convbin4_1 = self.conv_layer(self.poolbin3, 256, 512, "conv4_1", True)
        self.convbin4_2 = self.conv_layer(self.convbin4_1, 512, 512, "conv4_2", True)
        self.convbin4_3 = self.conv_layer(self.convbin4_2, 512, 512, "conv4_3", True)
        self.convbin4_4 = self.conv_layer(self.convbin4_3, 512, 512, "conv4_4", True)

        self.poolbin4 = self.max_pool(self.convbin4_4, "pool4")
        self.conv4_1 = self.conv_layer_full(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer_full(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer_full(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer_full(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, "pool4")
        self.loss4 = tf.reduce_mean(tf.square(self.pool4 - self.poolbin4))


        self.convbin5_1 = self.conv_layer(self.poolbin4, 512, 512, "conv5_1", True)
        self.convbin5_2 = self.conv_layer(self.convbin5_1, 512, 512, "conv5_2", True)
        self.convbin5_3 = self.conv_layer(self.convbin5_2, 512, 512, "conv5_3", True)
        self.convbin5_4 = self.conv_layer(self.convbin5_3, 512, 512, "conv5_4", True)
        self.poolbin5 = self.max_pool(self.convbin5_4, "pool5")

        self.conv5_1 = self.conv_layer_full(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer_full(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer_full(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer_full(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, "pool5")
        self.loss5 = tf.reduce_mean(tf.square(self.pool5 - self.poolbin5))

        with tf.variable_scope("fc6"):
            W_fc6 = self.weight_variable([512, 4096],  "fc6_weights")
            W_fc6 = self.quantize(W_fc6)
            x = tf.reshape(self.poolbin5, [-1, 512])
            self.fcbin6 = tf.matmul(x, W_fc6)

            self.fcbin6_bn = tf.contrib.layers.batch_norm(
                        self.fcbin6, decay = 0.99, scale=True, epsilon=BN_EPSILON, is_training=self.is_training)
            self.relubin6 = tf.nn.relu(self.fcbin6)

        self.relubin6 = tf.cond(self.is_training,
                lambda: tf.nn.dropout(self.relubin6, self.dropout),
                lambda: self.relubin6
        )

        self.fc6 = self.fc_layer_full(self.pool5, 512, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        self.loss6 = tf.reduce_mean(tf.square(self.relu6 - self.relubin6))



        self.fcbin7 = self.fc_layer(self.relubin6, 4096, 4096, "fc7")
        self.relubin7 = tf.nn.relu(self.fcbin7)

        self.relubin7 = tf.cond(self.is_training,
                lambda: tf.nn.dropout(self.relubin7, self.dropout),
                lambda: self.relubin7
        )


        self.fc7 = self.fc_layer_full(self.fc6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        self.loss7 = tf.reduce_mean(tf.square(self.relu7 - self.relubin7))

        with tf.variable_scope("fc8"):
            W_fc8 = self.weight_variable([4096, 100], "fc8_weights")
            W_fc8_bias = self.weight_variable([100], "fc8_biases")
            self.fc8 = tf.matmul(self.relubin7, W_fc8)

        self.fc8_full = self.fc_layer_full(self.relu7, 4096, 100, "fc8")
        self.loss8 = tf.reduce_mean(tf.square(self.fc8 - self.fc8_full))


        # self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name
        )

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name
        )

    def conv_layer(self, bottom, in_channels, out_channels, name, binary = False):
        with tf.variable_scope(name):
            filt, _ = self.get_conv_var(3, in_channels, out_channels, name)
            if binary == True:
                print(name, "is binary layer")
                filt = self.quantize(filt)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME")
            bias = conv

            bn_layer =tf.contrib.layers.batch_norm(
                        bias,  decay=0.999, scale=True, is_training=self.is_training)
            relu = tf.nn.relu(bn_layer)
            return relu


    def conv_layer_no_bn(self, bottom, in_channels, out_channels, name, binary = False):
        with tf.variable_scope(name):
            filt, _ = self.get_conv_var(3, in_channels, out_channels, name)
            if binary == True:
                print(name, "is binary layer")
                filt = self.quantize(filt)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME")
            bias = conv
            print(name, "without bn")
            ##虽然有计算，但结果无关
            bn_layer =tf.contrib.layers.batch_norm(
                        bias,  decay=0.999, scale=True, is_training=self.is_training)
            relu = tf.nn.relu(bn_layer)
            relu = tf.nn.relu(bias)
            return relu

    def conv_layer_no_bn(self, bottom, in_channels, out_channels, name, binary = False):
        with tf.variable_scope(name):
            filt, _ = self.get_conv_var(3, in_channels, out_channels, name)
            if binary == True:
                print(name, "is binary layer")
                filt = self.quantize(filt)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME")
            bias = conv
            print(name, "without bn")
            ##虽然有计算，但结果无关
            bn_layer =tf.contrib.layers.batch_norm(
                        bias,  decay=0.999, scale=True, is_training=self.is_training)
            relu = tf.nn.relu(bn_layer)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name, binary = False):
        with tf.variable_scope(name):
            weights, _ = self.get_fc_var(in_size, out_size, name)
            if binary == True:
                print(name, "is binary layer")
                weights = self.quantize(weights)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.matmul(x, weights)

            # fc_bn_layer = tf.contrib.layers.batch_norm(
            #             fc, decay = 0.99, scale=True, is_training=self.is_training)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels], 0.0, 0.001
        )
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
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
    def conv_layer_full(self, bottom, in_channels, out_channels, name):
        filt, conv_bias = self.get_constant_conv_var(3, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME")
        bias = tf.nn.bias_add(conv, conv_bias)
        beta, gamma = self.get_conv_bn(out_channels, name)
        bn_layer = self.batch_norm(bias, beta, gamma, self.is_training)
        relu = tf.nn.relu(bn_layer)
        return relu

    def fc_layer_full(self, bottom, in_size, out_size, name):
        x = tf.reshape(bottom, [-1, in_size])
        weights, biases  = self.get_constant_fc_var(in_size, out_size, name)
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        if(name == "fc8"):
            return fc
        else:
            beta, gamma = self.get_conv_bn(out_size, name)
            fc_bn_layer = self.batch_norm_fc(fc, beta, gamma, self.is_training)
            return fc_bn_layer



    def get_constant_conv_var(self, filter_size, in_channels, out_channels, name):
        filterShape = [filter_size, filter_size, in_channels, out_channels]
        biasShape = [out_channels]
        print("load file:", name)
        filters =  tf.constant(np.loadtxt('weights/' + name + '_' +  name + '_filters.txt', dtype=np.float32).reshape(filterShape))
        biases = tf.constant(np.loadtxt('weights/' + name + '_' + name + '_biases.txt', dtype=np.float32).reshape(biasShape))
        return filters, biases

    def get_conv_bn(self, size, name):
        shape = [size]
        beta = tf.constant(np.loadtxt('weights/' + name + '_BatchNorm_beta.txt', dtype=np.float32).reshape(shape))
        gamma = tf.constant(np.loadtxt('weights/' + name + '_BatchNorm_gamma.txt', dtype=np.float32).reshape(shape))

        return beta, gamma

    def batch_norm(self, x, beta,gamma,phase_train,scope='bn',decay=0.9,eps=BN_EPSILON):
        with tf.variable_scope(scope):
            batch_mean,batch_var = tf.nn.moments(x,[0,1,2],name='moments')
            normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, eps)

        return normed

    def get_constant_fc_var(self, in_size, out_size, name):
        weightShape = [in_size, out_size]
        biasShape = [out_size]
        print("load file: ", name)
        weights =  tf.constant(np.loadtxt('weights/' + name + '_' + name + '_weights.txt', dtype=np.float32).reshape(weightShape))
        biases = tf.constant(np.loadtxt('weights/' + name + '_' + name + '_biases.txt', dtype=np.float32).reshape(biasShape))

        return weights, biases

    def batch_norm_fc(self, x, beta,gamma,phase_train,scope='bn',decay=0.99,eps=BN_EPSILON):
        with tf.variable_scope(scope):
            batch_mean,batch_var = tf.nn.moments(x,[0],name='moments')
            normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, eps)

        return normed