#coding=utf-8
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np

import pickle

from utils import cifar100_read_data
from utils.args import parse_args


from binNet import Vgg19
args = parse_args()

# data_dir
data_dir = "../" + args.dataset

# log_dir = args.bin_model
weight_dir = args.non_bin_model

# batch size
batch_size = 10000
test_batch_size = 10000
# max_steps
max_steps = args.max_steps


# validate model every n steps
eval_every_n = args.eval_every_n

drop_out = 1.0
# learning rate
lr = 1e-7
# train phase
BN_TRAIN_PHASE = True
BN_TEST_PHASE = False


IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

ProbFile = "probBinary.txt"
TrainProbFile = "trainProbBinary.txt"
top5ProbFile = "top5ProbBinary.txt"
top5TrainProbFile = "top5TrainProbBinary.txt"

cifar100Data = cifar100_read_data.Cifar100DataReader(data_dir, batch_size, test_batch_size)

x = tf.placeholder(tf.float32, shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

y_ = tf.placeholder(tf.float32, shape=(None, 100))

train_phase = tf.placeholder(tf.bool, name="phase")
vgg = Vgg19(x, train_phase, drop_out)
y = vgg.fc8
y_full = vgg.fc8_full

prob = tf.nn.softmax(y)
max_value = tf.reduce_max(prob, reduction_indices=[1])

weight_decay = 5e-4
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_),
    name="cross_entroy_per_example",
)

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
loss = l2_loss * weight_decay + cross_entropy
# loss = loss * 5 + vgg.loss1*2 + vgg.loss2
# loss = loss*8 + vgg.loss1 + vgg.loss2 + vgg.loss3 + vgg.loss4 + vgg.loss5*2 + vgg.loss6*2 + vgg.loss7*3 + vgg.loss8*2
# loss = loss*8 + vgg.loss1*2 + vgg.loss2*4 + vgg.loss3 + vgg.loss4 + vgg.loss5 + vgg.loss6 + vgg.loss7*2 + vgg.loss8*3


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

correct_pred_full = tf.equal(tf.argmax(y_, 1), tf.argmax(y_full, 1))
accuracy_full = tf.reduce_mean(tf.cast(correct_pred_full, tf.float32), name="accuracy_full")

top5_acc = tf.nn.in_top_k(y, tf.argmax(y_, axis=1), 5)
top5_value = tf.nn.top_k(prob, 5).values
top5_acc = tf.reduce_mean(tf.cast(top5_acc, tf.float32), name="top5_accuracy")

# best_acc = 0.6079
save_baseline_acc = 0.65
best_top5 = 0.8748


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess, coord)
    print("Reading checkpoints...")
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    ckpt = tf.train.get_checkpoint_state(args.bin_model)
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

    print("start training&validating binary neural network....")
    try:

        for epoch in range(max_steps):
            image_batch, label_batch = cifar100Data.next_train_data()

            total_loss, train_acc, max_prediction_train, top5_value_prediction_train = sess.run(
                [loss, accuracy, max_value, top5_value], feed_dict={x: image_batch, y_: label_batch, train_phase: BN_TRAIN_PHASE}
            )


            if epoch % eval_every_n == 0:
                test_image_batch, test_label_batch =  cifar100Data.next_test_data()

                acc, prediction_loss, top5, acc_full, max_prediction, top5_value_prediction = sess.run(
                    [accuracy, loss, top5_acc, accuracy_full, max_value, top5_value], feed_dict={x: test_image_batch, y_: test_label_batch, train_phase: BN_TEST_PHASE}
                )
                if top5 > best_top5:
                    best_top5 = top5


                print("epoch: ",epoch, "; train set acc:", train_acc, "; full neural network acc: ", acc_full, "train set loss: ", total_loss)
                print("test set top5: ", top5, "; top1: ", acc, "test_loss:",prediction_loss, "; best top5: ", best_top5)
               
                print("======================================================================================================")
                # np.savetxt(str(TrainProbFile), max_prediction_train.flatten())
                # np.savetxt(str(ProbFile), max_prediction.flatten())

                np.savetxt(str(top5ProbFile), top5_value_prediction, delimiter=',')
                np.savetxt(str(top5TrainProbFile), top5_value_prediction_train, delimiter=',')
                print("......save result txt......")



                #modelPath = "target/"
                # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                #     weight = (sess.run([var]))[0].flatten().tolist()
                #     filename = (str(var).split())[1].replace('/', '_')
                #     print(filename)
                #     filename = filename.replace("'", "").replace(':0', '') + '.txt'

                #     # index = filename.find('_')
                #     # filename = filename[filename.find('_') + 1:]
                #     print("文件保存目录: ", modelPath)
                #     np.savetxt(str(modelPath + filename), weight)
                
                # print("......Ok......")
                



    except Exception as e:
        print('done!', e)
    finally:
        coord.request_stop()
        print("stop")

