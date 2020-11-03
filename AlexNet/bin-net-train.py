#coding=utf-8
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np

import pickle

from alexNet import AlexNet
from utils.args import parse_args
from utils import cifar100_read_data

args = parse_args()


# data_dir
data_dir = "../" + args.dataset
resize = args.resize

# log dir
log_dir = "bin-logs/"
weight_dir = "bin-weights/"

# batch size
batch_size = 4000
test_batch_size = 4000

# max_steps
max_steps = 50000


# validate model every n steps
eval_every_n = 5000
eval_every_best = 100

# learning rate
lr = 1e-6
# train phase
BN_TRAIN_PHASE = True
BN_TEST_PHASE = False

# drop out
drop_out = 1.0

IMAGE_WIDTH = 70
IMAGE_HEIGHT = 70

top5ProbFile = "top5Prob.txt"
top5TrainProbFile = "top5TrainProb.txt"

# cifar100Data = cifar10_read.Cifar100DataReader(data_dir, batch_size, test_batch_size)
cifar100Data = cifar100_read_data.Cifar100DataReader(data_dir, batch_size, test_batch_size, resize=resize)

x = tf.placeholder(tf.float32, shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

y_ = tf.placeholder(tf.float32, shape=(None, 100))

train_phase = tf.placeholder(tf.bool, name="phase")

alexnet = AlexNet(x, train_phase, drop_out)

y = alexnet.fcbin8
y_full = alexnet.fc8

prob = tf.nn.softmax(y)
max_value = tf.reduce_max(prob, reduction_indices=[1])

weight_decay = 5e-4
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_),
    name="cross_entroy_per_example",
)



loss = cross_entropy
# loss = loss*5 + alexnet.loss1 + alexnet.loss2*2 + alexnet.loss3*2 + alexnet.loss4 + alexnet.loss5 + alexnet.loss6 + alexnet.loss7 + alexnet.loss8*3

train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)



correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

correct_pred_full = tf.equal(tf.argmax(y_, 1), tf.argmax(y_full, 1))
accuracy_full = tf.reduce_mean(tf.cast(correct_pred_full, tf.float32), name="accuracy_full")

top5_acc = tf.nn.in_top_k(y, tf.argmax(y_, axis=1), 5)
top5_value = tf.nn.top_k(prob, 5).values
top5_acc = tf.reduce_mean(tf.cast(top5_acc, tf.float32), name="top5_accuracy")

# best_acc = 0.6079
best_acc = 0.42
best_top5 = 0.46

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess, coord)
    print("Reading checkpoints...")
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        for var_name in tf.contrib.framework.list_variables(ckpt.model_checkpoint_path):
            print(var_name)

        print("Loading success, global_step is %s" % global_step)
    else:
        print("No checkpoint file found")
        try:
            sess.run(tf.global_variables_initializer())
        except Exception as e:
            print(e)
        finally:
            print("Init done")

    # sess.run(tf.global_variables_initializer())

    print("start training....")
    try:

        for epoch in range(max_steps):
            image_batch, label_batch = cifar100Data.next_train_data()

            # __, total_loss, train_acc, max_prediction_train = sess.run(
            #     [train, loss, accuracy, max_value], feed_dict={x: image_batch, y_: label_batch, train_phase: BN_TRAIN_PHASE}
            # )

            total_loss, train_acc, max_prediction_train, top5_value_prediction_train = sess.run(
               [loss, accuracy, max_value, top5_value], feed_dict={x: image_batch, y_: label_batch, train_phase: BN_TRAIN_PHASE}
            )



            if epoch % eval_every_best == 0:
                test_image_batch, test_label_batch =  cifar100Data.next_test_data()

                acc, prediction_loss, top5, acc_full, max_prediction, top5_value_prediction = sess.run(
                    [accuracy, loss, top5_acc, accuracy_full, max_value, top5_value], feed_dict={x: test_image_batch, y_: test_label_batch, train_phase: BN_TEST_PHASE}
                )
                # acc, prediction_loss, top5, max_prediction, top5_value_prediction = sess.run(
                #     [accuracy, loss, top5_acc,  max_value, top5_value], feed_dict={x: test_image_batch, y_: test_label_batch, train_phase: BN_TEST_PHASE}
                # )


                print("epoch: ",epoch, "; train acc:", train_acc,  "train loss: ", total_loss, "; full acc: ", acc_full)
                print("top5: ", top5, "; top1: ", acc, "test_loss:",prediction_loss, "; best top5: ", best_top5)
                print("================================================================")


                np.savetxt(str(top5ProbFile), top5_value_prediction, delimiter=',')
                np.savetxt(str(top5TrainProbFile), top5_value_prediction_train, delimiter=',')
                print("save txt......")

                # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                #     weight = (sess.run([var]))[0].flatten().tolist()
                #     filename = (str(var).split())[1].replace('/', '_')
                #     filename = weight_dir + filename.replace("'", "").replace(':0', '') + '.txt'
                #     print("saving", filename)
                #     np.savetxt(str(filename), weight)

                # if epoch % eval_every_n == 0:
                #     print("save check points.......")
                #     saver = tf.train.Saver()
                #     saver.save(sess, './bin-logs/models_weighted-adam-conv4-conv5-fc8%s.ckpt' % acc)
                #     3

                # if top5 > best_top5:
                #     best_top5 = top5


                # if acc > best_acc:
                #     print("save check points.......")
                #     saver = tf.train.Saver()
                #     saver.save(sess, './bin-logs/models_weighted-adam-conv4-conv5-fc8%s.ckpt' % acc)

                #     best_acc = acc


    except Exception as e:
        print('done!', e)
    finally:
        coord.request_stop()
        print("stop")


print("hello")
