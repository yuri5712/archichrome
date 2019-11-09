import io
import os

from convolutional_autoencoder import Network
import tensorflow as tf
import convolutional_autoencoder
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import argparse
import clustering
from clustering import Cluster

# parser = argparse.ArgumentParser()
# parser.add_argument("model_dir", nargs='?', default=
# "C:/Users/yuriK/OneDrive/ドキュメント/#colour/code/Tensorflow-Segmentation-master/Tensorflow-Segmentation-master/save/C7,64,2C7,64,1M2C7,64,2C7,64,1M2C7,64,2C7,64,1M2/2018-08-09_035250"
# "", type=str, help="Path to directory storing checkpointed model.")  # best performance2018-08-09_035250
# parser.add_argument("test_image", nargs='?', default=os.path.join(resized_folder, streetviewName), type=str,
#                     help="Path to image for which the segmentation should be performed.")
# parser.add_argument("--out", default="/tmp", type=str,
#                     help="Path to directory to store resulting image.")  # if not list, define nargs as ?
# print(parser)
#
# args = parser.parse_args()
#
# test_image = args.test_image
# checkpoint = args.model_dir
#
# with tf.Session() as sess:
#     saver = tf.train.Saver(tf.global_variables())  # initially tf.all_variables()
#     ckpt = tf.train.get_checkpoint_state(checkpoint)
#     if ckpt and ckpt.model_checkpoint_path:
#         print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
#         saver.restore(sess, ckpt.model_checkpoint_path)
#     else:
#         raise IOError('No model found in {}.'.format(checkpoint))
#
#     image = cv2.imread(test_image, 1)  # load grayscale* changed to color scale because n>0 = 1
#
#     # remain image color before segmentation as rgb_image (converted bgr to rgb)
#     bgr_image = image
#     rgb_image = bgr_image[:, :, [2, 1, 0]]
#
#     # Resize input image
#     # もとの写真を正方形にcropする
#     inh, inw, inc = image.shape
#     print("inh, inw, inc: ", inh, inw, inc)
#     if (inh != inw):
#         print("image shape: inh != inw")
#         if (inh <= inw):
#             image = image[20:20 + inh, 0:inh]  # y1:y2, x1:x2
#         else:
#             image = image[20:20 + inw, 0:inw]
#         image = cv2.resize(image, (network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
#         cv2.imwrite(os.path.join('input_resized.jpg'), image)
#
#     image = np.array(image)
#     image = np.multiply(image, 1.0 / 255)
#
#     segmentation = sess.run(network.segmentation_result, feed_dict={
#         network.inputs: np.reshape(image, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS])})
#
#     print("segmentation_len: ", len(segmentation))
#     segNormalized = segmentation[0]
#     segmented_image = np.dot(segmentation[0], 255)
#
#
# test_inputs, test_targets = dataset.test_set
#
#                     test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
#                     test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
#                     test_inputs = np.multiply(test_inputs, 1.0 / 255)
#
#                     print(test_inputs.shape)
#                     summary, test_accuracy = sess.run([network.summaries, network.accuracy],
#                                                       feed_dict={network.inputs: test_inputs,
#                                                                  network.targets: test_targets,
#                                                                  network.is_training: False})
#

folder_path = "resized_CMP"
inputs = os.listdir(os.path.join(folder_path, "inputs"))
targets = os.listdir(os.path.join(folder_path, "targets"))
input_path = os.path.join(folder_path, "inputs", inputs[0])
target_path = os.path.join(folder_path, "targets", targets[0])
test_data = cv2.imread(input_path, 1)
test_label = cv2.imread(target_path, 1)

print("input_path", input_path)
print("target_path", target_path)

tf.reset_default_graph()
layers = []
layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True))

layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True))

layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))
layers.append(MaxPool2d(kernel_size=2, name='max_3'))

network = convolutional_autoencoder.Network(layers)

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", nargs='?', default=
r"C:/Users/yuriK/OneDrive/ドキュメント/#colour/code/Tensorflow-Segmentation-master/Tensorflow-Segmentation-master/save/C7,64,2C7,64,1M2C7,64,2C7,64,1M2C7,64,2C7,64,1M2/2018-08-09_035250"
"", type=str, help="Path to directory storing checkpointed model.")# best performance2018-08-09_035250
parser.add_argument("test_image", nargs='?', default=os.path.join(folder_path, "inputs"), type=str, help="Path to image for which the segmentation should be performed.")
parser.add_argument("--out", default="/tmp", type=str, help="Path to directory to store resulting image.")#if not list, define nargs as ?
print(parser)
args = parser.parse_args()
# test_image = args.test_image
checkpoint = args.model_dir

v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])
# saver = tf.train.Saver(tf.global_variables())
saver = tf.train.Saver()

tf.reset_default_graph()
with tf.Session() as sess:
    # ckpt_path = tf.train.latest_checkpoint(r'C:/Users/yuriK/OneDrive/ドキュメント/#colour/code/Tensorflow-Segmentation-master/Tensorflow-Segmentation-master/save/C7,64,2C7,64,1M2C7,64,2C7,64,1M2C7,64,2C7,64,1M2/2018-08-09_035250/model.ckpt-600.data-00000-of-00001')
    # saver = ""
    saver = tf.train.Saver(tf.global_variables())  # initially tf.all_variables()
    ckpt = tf.train.get_checkpoint_state(checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise IOError('No model found in {}.'.format(checkpoint))
    # saver = saver.restore(sess, checkpoint)
    # ckpt = tf.train.get_checkpoint_state(checkpoint)
    res = sess.run(network.accuracy, feed_dict={
        network.x: test_data,
        network.y: test_label,
        network.is_training: False
    })
print('accuracy: ', res)









if __name__ == '__main__':

    print ("done")