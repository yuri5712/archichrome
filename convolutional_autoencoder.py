import sys
sys.path.append('../')
import math
import os
import time
from math import ceil

import cv2
import matplotlib

matplotlib.use('Agg')
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from imgaug import augmenters as iaa
from imgaug import imgaug

from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io

np.set_printoptions(threshold=np.nan)


# @ops.RegisterGradient("MaxPoolWithArgmax")
# def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
#     return gen_nn_ops._max_pool_grad(op.inputs[0],
#                                      op.outputs[0],
#                                      grad,
#                                      op.get_attr("ksize"),
#                                      op.get_attr("strides"),
#                                      padding=op.get_attr("padding"),
#                                      data_format='NHWC')


class Network:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 3

    def __init__(self, layers=None, per_image_standardization=True, batch_norm=True, skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).

        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_3'))

        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS], name='targets')#image channels originally =1
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        if per_image_standardization:
            list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
            net = tf.stack(list_of_images_norm)
        else:
            net = self.inputs

        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        # print("Current input shape: ", net.get_shape())

        layers.reverse()
        Conv2d.reverse_global_variables()

        # DECODER
        for layer in layers:
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        self.segmentation_result = tf.sigmoid(net)

        # segmentation_as_classes = tf.reshape(self.y, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH, 1])
        # targets_as_classes = tf.reshape(self.targets, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        # print(self.y.get_shape())
        # self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(segmentation_as_classes, targets_as_classes))
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        # MSE loss
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()


class Dataset:
    def __init__(self, batch_size, folder="CMP_facade_DB_base"):

        self.batch_size = batch_size

        print("os.listdir(os.path.join(folder, 'targets_original')): ", os.listdir(os.path.join(folder, 'targets_original')))########
        train_files_input, validation_files_input, test_files_input = self.train_valid_test_split(os.listdir(os.path.join(folder, 'inputs_original')))#######
        train_files_target, validation_files_target, test_files_target = self.train_valid_test_split(os.listdir(os.path.join(folder, 'targets_original')))#########

        self.train_inputs, self.train_targets  = self.file_paths_to_images(folder, train_files_input, train_files_target)

        self.test_inputs, self.test_targets = self.file_paths_to_images(folder, test_files_input, test_files_target, True)

        self.pointer = 0


    #一枚ずつリサイズする
    def resize_images(self, input_image, target_image, i):
        # Create 250*250 Size Image in new dir
        # used inside def file_paths_to_images
        # resized train input = rtinput

        #resizeする統一の大きさを決める(250*250)
        size = (Network.IMAGE_HEIGHT, Network.IMAGE_WIDTH)

        #もとの写真を正方形にcropする
        inh, inw, inc = input_image.shape
        tah, taw, tac = target_image.shape
        print("inh, inw, inc: ", inh, inw, inc)
        print("tah, taw, tac: ", tah, taw, tac)

        if(inh == tah and inw == taw and inh != inw):
            print("image shape: inh == tah and inw == taw and inh != inw")
            # if(inh <= inw):
            #     input_image = input_image[inh:inh, 0:0]
            #     target_image = target_image[inh:inh, 0:0]
            # else:
            #     input_image = input_image[inw:inw, 0:0]
            #     target_image = target_image[inw:inw, 0:0]
            if (inh <= inw):
                input_image = input_image[0:inh, 0:inh]
                target_image = target_image[0:inh, 0:inh]
            else:
                input_image = input_image[0:inw, 0:inw]
                target_image = target_image[0:inw, 0:inw]
            rtinput = cv2.resize(input_image, size)
            rttarget = cv2.resize(target_image, size)
            # print("len(rtinput[0]", len(rtinput[0]))
            # print("len(rtinput): ", len(rtinput))
            # print("len(rttarget): ", len(rttarget))

            cv2.imwrite(os.path.join('resized_CMP', 'inputs', 'input' + str(i) + '.jpg'), rtinput)
            cv2.imwrite(os.path.join('resized_CMP', 'targets', 'target' + str(i) + '.png'), rttarget)

        else:
            print("image shape: cannot be reshaped or already is rectangle")
            print("input_image: ", input_image)
            rtinput = input_image
            rttarget = target_image
            cv2.imwrite(os.path.join('resized_CMP\inputs', 'input' + str(i) + '.jpg'), rtinput)
            cv2.imwrite(os.path.join('resized_CMP\targets', 'target' + str(i) + '.png'), rttarget)

        return rtinput, rttarget

    #inputsフォルダとtargetsフォルダの中のイメージを読み込んでくる。def resize_imageを呼ぶ。
    def file_paths_to_images(self, folder, train_files_input, train_files_target, verbose=False):
        inputs = []
        targets = []
        resized_train_input = []
        resized_train_target = []
        i = 0

        for file_input, file_target in zip(train_files_input, train_files_target):
            if (file_input[-4:] == '.jpg' and file_target[-4:] == '.png'):
                print(file_input, file_target)

                input_image = os.path.join(folder, 'extended_inputs', file_input)
                target_image = os.path.join(folder, 'extended_targets', file_target)

                # print("input_image: ", input_image)
                # print("target_image: ", target_image)

                input_image = np.array(cv2.imread(input_image, 1))  # load color 3channels>0, grayscale=0, image itself<0
                #Consider if above line needed
                inputs.append(input_image)

                target_image = cv2.imread(target_image, 1)  # load color 3channels<0, grayscale=0, image itself<0
                #target_image = cv2.threshold(target_image, 127, 1, cv2.THRESH_BINARY)[1] #target image pixcels assigned to black or white
                #Consider if 3 RGB channels need division
                targets.append(target_image)

                #print("input_image: ", input_image)
                # print("target_image: ", target_image)

                rtinput, rttarget = self.resize_images(input_image, target_image, i)
                rtinput_image = np.multiply(rtinput, 1.0 / 255)
                rttarget_image = np.multiply(rttarget, 1.0 / 255)
                resized_train_input.append(rtinput_image)
                resized_train_target.append(rttarget_image)

                i = i+1


            else:
                print("this is not an image file: ", file_input[-4:], file_target[-4:])

        return resized_train_input, resized_train_target


    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, 0.15, 0.15)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        batch_inputs = []
        batch_targets = []
        for j in range(self.batch_size):
            #print("self.pointer", self.pointer)
            batch_inputs.append(np.array(self.train_inputs[self.pointer + j]))
            batch_targets.append(np.array(self.train_targets[self.pointer + j]))

        self.pointer += self.batch_size

        return np.array(batch_inputs, dtype=np.uint8), np.array(batch_targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)

def remap255(self, list):
    mx = max(list)
    mn = min(list)
    remap_list = np.array([(float(x-mn)/float(mx))*255 for x in list])
    return  remap_list

def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num):
    n_examples_to_plot = 12
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        print("test_inputs[example_i].flatten(): ", test_inputs[example_i].flatten(), "flatten")
        image_value = remap255([test_inputs[example_i].flatten()])
        axs[0][example_i].imshow(test_inputs[example_i])
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32))

        #print(test_segmentation[example_i])

        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]))
        # test_image_thresholded = np.array(
        #     [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
        test_image_thresholded = np.array(
            [x*255 for x in test_segmentation[example_i].flatten()])

        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]))

    buf = io.BytesIO()
    print(buf)
    cv2.imwrite(buf)
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf


def train():
    BATCH_SIZE = 100#100  # 32, 64, 128, 256, 512

    network = Network()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # create directory for saving models
    os.makedirs(os.path.join('save', network.description, timestamp))
    # len(rtinput[0])
    dataset = Dataset(folder=".\CMP_facade_DB_base", batch_size=BATCH_SIZE)
    #was originally 'data{}_{}'.format(network.IMAGE_HEIGHT, network.IMAGE_WIDTH), batch_size=BATCH_SIZE

    inputs, targets = dataset.next_batch()
    print("inputs.shape: ", inputs.shape, "targets.shape: ", targets.shape)

    # augmentation_seq = iaa.Sequential([
    #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    #     iaa.GaussianBlur(sigma=(0, 2.0))  # blur images with a sigma of 0 to 3.0
    # ])

    augmentation_seq = iaa.Sequential([
        iaa.Crop(px=(0, 16), name="Cropper"),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5, name="Flipper"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.Dropout(0.02, name="Dropout"),
        iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="GaussianNoise"),
        iaa.Affine(translate_px={"x": (-network.IMAGE_HEIGHT // 3, network.IMAGE_WIDTH // 3)}, name="Affine")
    ])

    # change the activated augmenters for binary masks,
    # we only want to execute horizontal crop, flip and affine transformation
    def activator_binmasks(images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "Dropout", "GaussianNoise"]:
            return False
        else:
            # default value for all other augmenters
            return default

    hooks_binmasks = imgaug.HooksImages(activator=activator_binmasks)

    time_axis = []
    batch_axis = []
    accuracy_axis = []
    cost_axis = []
    # time_axis = np.array([])
    # batch_axis = np.array([])
    # accuracy_axis = np.array([])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
                                               graph=tf.get_default_graph())
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

        test_accuracies = []
        # Fit all training data
        n_epochs = 500#500
        global_start = time.time()
        for epoch_i in range(n_epochs):
            dataset.reset_batch_pointer()

            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

                augmentation_seq_deterministic = augmentation_seq.to_deterministic()

                start = time.time()
                start_time = time.clock()
                batch_inputs, batch_targets = dataset.next_batch()
                batch_inputs = np.reshape(batch_inputs,
                                          (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
                batch_targets = np.reshape(batch_targets,
                                           (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))

                batch_inputs = augmentation_seq_deterministic.augment_images(batch_inputs)
                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                batch_targets = augmentation_seq_deterministic.augment_images(batch_targets, hooks=hooks_binmasks)

                cost, _ = sess.run([network.cost, network.train_op],
                                   feed_dict={network.inputs: batch_inputs, network.targets: batch_targets,
                                              network.is_training: True})
                end = time.time()
                print('{}/{}, epoch: {}, cost: {}, batch time: {}, start time: {}, time:{}'.format(batch_num,
                                                                          n_epochs * dataset.num_batches_in_epoch(),
                                                                          epoch_i, cost, end - start, start_time, datetime.datetime.today()))

                if batch_num % 100 == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch(): # batch_num % 100 ==0
                    test_inputs, test_targets = dataset.test_set
                    # test_inputs, test_targets = test_inputs[:100], test_targets[:100]

                    test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
                    test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
                    test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    print(test_inputs.shape)
                    summary, test_accuracy = sess.run([network.summaries, network.accuracy],
                                                      feed_dict={network.inputs: test_inputs,
                                                                 network.targets: test_targets,
                                                                 network.is_training: False})

                    summary_writer.add_summary(summary, batch_num)

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    test_accuracies.append((test_accuracy, batch_num))
                    print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    max_acc = max(test_accuracies)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))

                    # Plot example reconstructions
                    n_examples = 12
                    test_inputs, test_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]
                    test_inputs = np.multiply(test_inputs, 1.0 / 255)

                    test_segmentation = sess.run(network.segmentation_result, feed_dict={
                        network.inputs: np.reshape(test_inputs,
                                                   [n_examples, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS])})

                    cost_axis.append(cost)
                    accuracy_axis.append(test_accuracy)
                    batch_axis.append(batch_num)
                    time_axis.append(start_time)

                    #
                    # #draw_result関数を使って学習の表をつくる。ここでは必要なし
                    # # Prepare the plot
                    # test_plot_buf = draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network,
                    #                              batch_num)
                    #
                    # # Convert PNG buffer to TF image
                    # image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
                    #
                    # # Add the batch dimension
                    # image = tf.expand_dims(image, 0)
                    #
                    # # Add image summary
                    # image_summary_op = tf.summary.image("plot", image)
                    #
                    # image_summary = sess.run(image_summary_op)
                    # summary_writer.add_summary(image_summary)
                    #

                    if test_accuracy >= max_acc[0]:
                    # if test_accuracy >= 0.9:

                        checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=batch_num)
    # print("accuracy_axis:", accuracy_axis, "accuracy num:", len(accuracy_axis))
    # print("time_axis:", time_axis, "time num:", len(time_axis))
    # plt.plot(time_axis, accuracy_axis)
    # plt.title("BATCH_SIZE:" + str(BATCH_SIZE) + " n_epochs:" + str(n_epochs) + "Date: " + str(datetime.datetime.today()))
    # plt.xlabel("Time")
    # plt.ylabel("Accuracy")
    # plot_num = 0
    # for x,y in zip(time_axis, accuracy_axis):
    #     if plot_num % 2 ==0:
    #         plt.text(x, y-0.05, str(y))
    #     else:
    #         plt.text(x, y + 0.05, str(y))
    #     plot_num += 1
    # plt.ylim((0.0, 1.0))
    # plt.grid(True)
    # plt.show()

    print("loss_axis:", cost_axis, "cost num:", len(cost_axis))
    print("time_axis:", time_axis, "time num:", len(time_axis))
    plt.plot(time_axis, cost_axis)
    plt.title(
        "BATCH_SIZE:" + str(BATCH_SIZE) + " n_epochs:" + str(n_epochs) + "Date: " + str(datetime.datetime.today()))
    plt.xlabel("Time")
    plt.ylabel("Cost")
    # plot_num = 0
    # for x, y in zip(time_axis, cost_axis):
    #     if plot_num % 2 == 0:
    #         plt.text(x, y - 0.05, str(y))
    #     else:
    #         plt.text(x, y + 0.05, str(y))
    #     plot_num += 1
    plt.ylim((0.0, 1.0))
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    start = datetime.datetime.today()
    print('training start time:', start)

    train()

    print("##############################################################################")
    end = datetime.datetime.today()
    print('end time:', end)