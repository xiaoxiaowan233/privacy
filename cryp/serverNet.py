import tensorflow as tf

import numpy as np
from functools import reduce
import time

batch_norm_dir = "../inference/"

VGG_MEAN = [103.939, 116.779, 123.68]
BN_EPSILON = 1e-4

@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, x):

        self.G = tf.get_default_graph()
        self.rgb = x
        self.is_training = False
        self.build()
        self.loadFile = 0




    def build(self, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        ## conv1

        # self.convbin1_1,_,_ = self.conv_layer(self.rgb, 3, 64, "conv1_1", True)
        # self.convbin1_2,_,_  = self.conv_layer(self.convbin1_1, 64, 64, "conv1_2", True)
        # self.poolbin1 = self.max_pool(self.convbin1_2, "pool1")


        # with tf.variable_scope("bin"):
        # self.convbin2_1,_,_  = self.conv_layer(self.poolbin1, 64, 128, "conv2_1", True)
        # self.convbin2_2,_,_  = self.conv_layer(self.convbin2_1, 128, 128, "conv2_2", True)
        # self.poolbin2 = self.max_pool(self.convbin2_2, "pool2")


        # with tf.variable_scope("bin"):
        # self.convbin3_1,_,_ = self.conv_layer(self.poolbin2, 128, 256, "conv3_1", True)
        # self.convbin3_2,_,_ = self.conv_layer(self.convbin3_1, 256, 256, "conv3_2",True)
        # self.convbin3_3,_,_ = self.conv_layer(self.convbin3_2, 256, 256, "conv3_3", True)
        # self.convbin3_4,_,_ = self.conv_layer(self.convbin3_3, 256, 256, "conv3_4", True)
        # self.poolbin3 = self.max_pool(self.convbin3_4, "pool3")

        # with tf.variable_scope("bin"):
        # self.convbin4_1,_,_ = self.conv_layer(self.poolbin3, 256, 512, "conv4_1", True)
        # self.convbin4_2 = self.conv_layer(self.rgb, 512, 512, "conv4_2", True)
        # self.convbin4_3 = self.conv_layer(self.convbin4_2, 512, 512, "conv4_3", True)
        # self.convbin4_4 = self.conv_layer(self.convbin4_3, 512, 512, "conv4_4", True)
        # self.poolbin4 = self.max_pool(self.convbin4_4, "pool4")


        # with tf.variable_scope("bin"):
        # self.convbin5_1 = self.conv_layer(self.poolbin4, 512, 512, "conv5_1", True)
        # self.convbin5_2 = self.conv_layer(self.convbin5_1, 512, 512, "conv5_2", True)
        # self.convbin5_3 = self.conv_layer_no_bn(self.convbin5_2, 512, 512, "conv5_3", True)
        # self.convbin5_4 = self.conv_layer_no_bn(self.convbin5_3, 512, 512, "conv5_4", True)
        # self.poolbin5 = self.max_pool(self.convbin5_4, "pool5")

        # with tf.variable_scope("bin"):
        # self.fcbin6 = self.fc_layer(self.poolbin5, 512, 4096, "fc6")
        # self.relubin6 = tf.nn.relu(self.fcbin6)


        self.fcbin7 = self.fc_layer(self.rgb, 4096, 4096, "fc7")
        self.relubin7 = tf.nn.relu(self.fcbin7)
        self.fcbin8 = self.fc_layer(self.relubin7, 4096, 100, "fc8")


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name
        )

    ## binary: 是否做二值化处理

    def conv_layer(self, bottom, in_channels, out_channels, name, binary=False):

        with tf.variable_scope(name):
            # if freeze == False: 
            #     filt, conv_bias = self.get_conv_var(3, in_channels, out_channels, name)
            # else:

            filt = self.get_constant_conv_var(3, in_channels, out_channels, name)
            print("%s load weights from file" % name)
            # self.loadFile = self.loadFile + time.time() - t1

            if binary == True:
                print("%s is binary layer" % name)
                filt = self.quantize(filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME")

            beta, gamma = self.get_conv_bn(out_channels, name)
            bn_layer = self.batch_norm(conv, beta, gamma,name,out_channels, False)
            relu = tf.nn.relu(bn_layer)
            return relu

    def conv_layer_no_bn(self, bottom, in_channels, out_channels, name, binary=False):

        with tf.variable_scope(name):
            # if freeze == False: 
            #     filt, conv_bias = self.get_conv_var(3, in_channels, out_channels, name)
            # else:
            t1 = time.time()

            filt = self.get_constant_conv_var(3, in_channels, out_channels, name)
            print("%s load weights from file" % name)
            # self.loadFile = self.loadFile + time.time() - t1

            if binary == True:
                print("%s is binary layer" % name)
                filt = self.quantize(filt)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding="SAME")
            relu = tf.nn.relu(conv)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name, binary=False):
        with tf.variable_scope(name):

            t1 = time.time()
            weights  = self.get_constant_fc_var(in_size, out_size, name)
            print("%s load weights from file" % name)
            # self.loadFile = self.loadFile + time.time() - t1

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.matmul(x, weights)
            # beta, gamma = self.get_conv_bn(out_size, name)
            # bn_layer = self.batch_norm_fc(fc, beta, gamma, False)
            # normed = tf.layers.batch_normalization(x,center=True,scale=True,  beta_initializer = beta,  gamma_regularizer = gamma,training=False,)

            return fc

    def get_constant_conv_var(self, filter_size, in_channels, out_channels, name):
        filterShape = [filter_size, filter_size, in_channels, out_channels]
        filters =  tf.constant(np.loadtxt(batch_norm_dir + name + '_' +  name + '_filters.txt', dtype=np.float32).reshape(filterShape))
   
        return filters

    def get_constant_fc_var(self, in_size, out_size, name):
        weightShape = [in_size, out_size]
        weights =  tf.constant(np.loadtxt(batch_norm_dir + name + '_' + name + '_weights.txt', dtype=np.float32).reshape(weightShape))

        return weights



    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)

    def batch_norm(self, x, beta,gamma,name, out_channels, phase_train,scope='bn',decay=0.9,eps=BN_EPSILON):
        with tf.variable_scope(scope):
            batch_mean = tf.constant(np.loadtxt(batch_norm_dir + name + '_BatchNorm_mean.txt', dtype=np.float32).reshape(out_channels))
            batch_var = tf.constant(np.loadtxt(batch_norm_dir + name + '_BatchNorm_var.txt', dtype=np.float32).reshape(out_channels))

            normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, eps)

        return normed

    def batch_norm_fc(self, x, beta,gamma,phase_train,scope='bn',decay=0.9,eps=BN_EPSILON):
        with tf.variable_scope(scope):
            batch_mean,batch_var = tf.nn.moments(x, list(range(len(x.get_shape()) - 1)),name='moments')
            # batch_mean,batch_var = tf.nn.moments(x, tf.shape(x)[:-1],name='moments')
            normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, eps)

        return normed

    def get_conv_bn(self, size, name):
        shape = [size]
        t1 = time.time()
        beta = tf.constant(np.loadtxt(batch_norm_dir + name + '_BatchNorm_beta.txt', dtype=np.float32).reshape(shape))
        gamma = tf.constant(np.loadtxt(batch_norm_dir + name + '_BatchNorm_gamma.txt', dtype=np.float32).reshape(shape))
        # self.loadFile = self.loadFile + time.time() - t1

        return beta, gamma

    def hard_sigmoid(self, x):
        return tf.clip_by_value((x + 1.) / 2, 0, 1)

    def binary_tanh_unit(self, x):
        return 2 * self.hard_sigmoid(x) - 1
