import tensorflow as tf
import numpy as np

import ngraph_bridge
import os
import time

from serverNet import Vgg19
HEDataFile = "data.txt"
HELabelFile = "groundTruth.txt"
HEResultFile = "result.txt"


WIDTH = 13
HEIGHT = 13
DEPTH = 5

batch_size = 2048
shape = (batch_size, 4096)

x = tf.placeholder(tf.float32, [batch_size, 4096])



vgg = Vgg19(x)
prediction = vgg.fcbin8

x_image = np.loadtxt(HEDataFile, dtype=np.float32).reshape(shape)
x_image = x_image[:batch_size]


###明文看精度
#ground_truth = np.loadtxt(HELabelFile, dtype = np.float32).reshape(batch_size,100)

#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ground_truth, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy") 



with tf.Session() as sess:
    t = time.time()
    result = sess.run(prediction, feed_dict={x: x_image})
    print("time: ", time.time() - t)


    #According to configuration Settings
    
    # print("result: ", result.shape)
    # np.savetxt(str(HEResultFile), result.flatten())


    #acc = sess.run(accuracy, feed_dict={x: x_image})
    #print("accuracy: ", acc)

    # y_label_batch = np.argmax(ground_truth, 1)
    # y_pred = np.argmax(result, 1)
    # correct_prediction = np.equal(y_pred, y_label_batch)
    # error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    # test_accuracy = np.mean(correct_prediction)

    # print('Error count', error_count, 'of', batch_size, 'elements.')
    # print('Accuracy: %g ' % test_accuracy)
