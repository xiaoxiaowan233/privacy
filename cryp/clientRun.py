#coding=utf-8
import random
# random.seed(10086)
import tensorflow as tf
import numpy as np
# np.random.seed(10010)
import pickle
import sys
sys.path.append("..")
from utils import cifar100_read_data

from clientNet import Vgg19
import time

# data_dir
data_dir = "../cifar-100-python"


HEDataFile = "data.txt"
HELabelFile = "groundTruth.txt"

weight_dir = "../inference/"

# batch size
batch_size = 256
test_batch_size = 1024


IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32


cifar100Data = cifar100_read_data.Cifar100DataReader(data_dir, batch_size, test_batch_size)

x = tf.placeholder(tf.float32, shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

y_ = tf.placeholder(tf.float32, shape=(None, 100))

vgg = Vgg19(x)
y = vgg.fcbin8

correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


with tf.Session() as sess:
    print("start testing....")

    try:
        ##According to configuration Settings
        test_image_batch, test_label_batch =  cifar100Data.next_test_data()
        t = time.time()

        # result,conv5_2_mean_run,conv5_2_var_run,conv5_1_mean_run,conv5_1_var_run,conv4_4_mean_run,conv4_4_var_run,conv4_3_mean_run,conv4_3_var_run,conv4_2_mean_run,conv4_2_var_run,acc = \
        #             sess.run([y, conv5_2_mean, conv5_2_var, conv5_1_mean, conv5_1_var, conv4_4_mean, conv4_4_var, conv4_3_mean, conv4_3_var, conv4_2_mean, conv4_2_var,accuracy], feed_dict={x: test_image_batch, y_: test_label_batch})

        result = sess.run(y, feed_dict={x: test_image_batch, y_: test_label_batch})



        # runTime = time.time() - t
        # print("saving: ", result.shape)
        np.savetxt(str(HEDataFile), result.flatten())
        np.savetxt(str(HELabelFile), test_label_batch)
        # filename = weight_dir + "conv4_2_BatchNorm_mean.txt"
        # np.savetxt(str(filename), conv4_2_mean_run.flatten())
        # filename = weight_dir + "conv4_2_BatchNorm_var.txt"
        # np.savetxt(str(filename), conv4_2_var_run.flatten())

        # filename = weight_dir + "conv4_3_BatchNorm_mean.txt"
        # np.savetxt(str(filename), conv4_3_mean_run.flatten())
        # filename = weight_dir + "conv4_3_BatchNorm_var.txt"
        # np.savetxt(str(filename), conv4_3_var_run.flatten())

        # filename = weight_dir + "conv4_4_BatchNorm_mean.txt"
        # np.savetxt(str(filename), conv4_4_mean_run.flatten())
        # filename = weight_dir + "conv4_4_BatchNorm_var.txt"
        # np.savetxt(str(filename), conv4_4_var_run.flatten())

        # filename = weight_dir + "conv5_1_BatchNorm_mean.txt"
        # np.savetxt(str(filename), conv5_1_mean_run.flatten())
        # filename = weight_dir + "conv5_1_BatchNorm_var.txt"
        # np.savetxt(str(filename), conv5_1_var_run.flatten())

        # filename = weight_dir + "conv5_2_BatchNorm_mean.txt"
        # np.savetxt(str(filename), conv5_2_mean_run.flatten())
        # filename = weight_dir + "conv5_2_BatchNorm_var.txt"
        # np.savetxt(str(filename), conv5_2_var_run.flatten())

        # print("runTime: ", runTime)

        acc = sess.run(accuracy, feed_dict={x: test_image_batch, y_: test_label_batch})
        print("acc: ", acc)

    except Exception as e:
        print('done!', e)
    finally:
        # coord.request_stop()
        print("stop")

    # Save Model

print("hey")
                   