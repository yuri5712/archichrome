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


class Test:
    def __init__(self, folder="DLimgs", resized_folder="streetview", segmented_folder="segmented"):
        print("##############################################################################")
        self.dlImages = self.file_paths_to_DLimages(folder, os.listdir(os.path.join(folder)), resized_folder)

        self.label_color = self.inferDLimgs(resized_folder, os.listdir(os.path.join(resized_folder)), segmented_folder)

    # inputsフォルダとtargetsフォルダの中のイメージを読み込んでくる。def resize_imageを呼ぶ。
    def file_paths_to_DLimages(self, folder, imgs, resized_folder,  verbose=False, ):
        inputs = []
        targets = []
        resized_train_input = []
        resized_train_target = []

        streetviews = []
        resized_streetviews = []
        i = 0

        for img in imgs:
            print("img:", img)
            if (img[-4:] == '.jpg' or img[-5:] == '.jpeg'):
                # print(img[-4:], img[-4:])
                streetview = os.path.join(folder, img)
                # print("streetview", streetview)

                streetview = np.array(cv2.imread(streetview, 1))  # load color 3channels>0, grayscale=0, image itself<0
                # print("streetview", streetview)
                # Consider if above line needed
                streetviews.append(streetview)
                resized_streetview = self.resize_images(streetview, i, resized_folder)
                resized_streetview = np.multiply(resized_streetview, 1.0 / 255)
                resized_streetviews.append(resized_streetview)
                i = i + 1
            else:
                print("this is not an image file: ", img[-4:])

        # return resized_streetviews

    def resize_images(self, streetview, i, resized_folder):
        # Create 250*250 Size Image in new dir
        # used inside def file_paths_to_images
        # resized train input = rtinput

        # resizeする統一の大きさを決める(250*250)
        size = (Network.IMAGE_HEIGHT, Network.IMAGE_WIDTH)

        # もとの写真を正方形にcropする
        inh, inw, inc = streetview.shape
        # print("inh, inw, inc: ", inh, inw, inc)

        if (inh != inw):
            # print("image shape: inh == tah and inw == taw and inh != inw")
            if (inh <= inw):
                streetview = streetview[0:inh, 0:inh]
            else:
                streetview = streetview[0:inw, 0:inw]
            resized_streetview = cv2.resize(streetview, size)

            cv2.imwrite(os.path.join(resized_folder, 'rsv' + str(i) + '.jpg'), streetview)

        else:
            print("image shape: cannot be reshaped or already is rectangle")
            print("streetview: ", streetview)
            resized_streetview = streetview
            cv2.imwrite(os.path.join(resized_folder, 'rsv' + str(i) + '.jpg'), streetview)

        return streetview

    def inferDLimgs(self, resized_folder, rimgs, segmented_folder):

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

        no_label = []
        background = []
        facade = []
        window = []
        door = []
        cornice = []
        sill = []
        balcony = []
        blind = []
        deco = []
        molding = []
        pillar = []
        shop = []

        num = 0

        elementsNames = ["no_label", "background", "facade", "window", "door", "cornice", "sill", "balcony", "blind",
                         "deco", "molding", "pillar", "shop"]
        label_color = [no_label, background, facade, window, door, cornice, sill, balcony, blind, deco, molding, pillar,
                       shop]

        for r, rimg in enumerate(rimgs):
            print("##############################################################################")
            print("infer:", rimg)
            label_image, rgb_image = self.segmentationStreetview(rimg, network, resized_folder, segmented_folder)

            for label_clm, rgb_clm in zip(label_image, rgb_image):
                for label, rgb in zip(label_clm, rgb_clm):
                    if (label == 'background'):
                        background.append(rgb)
                    elif (label == 'facade'):
                        facade.append(rgb)
                    elif (label == 'window'):
                        window.append(rgb)
                    elif (label == 'door'):
                        door.append(rgb)
                    elif (label == 'cornice'):
                        cornice.append(rgb)
                    elif (label == 'sill'):
                        sill.append(rgb)
                    elif (label == 'balcony'):
                        balcony.append(rgb)
                    elif (label == 'blind'):
                        blind.append(rgb)
                    elif (label == 'deco'):
                        deco.append(rgb)
                    elif (label == 'molding'):
                        molding.append(rgb)
                    elif (label == 'pillar'):
                        pillar.append(rgb)
                    elif (label == 'shop'):
                        shop.append(rgb)
                    elif (label == 'no_label'):
                        no_label.append(rgb)

            # for ele in elementsNames:
            #     maskingImage = rgb_image
            #     for i,label_clm in enumerate(label_image):
            #         for j,label in enumerate(label_clm):
            #             if (label != ele):
            #                 maskingImage[i][j] = [255,255,255]
            #     print('maskingImage:', maskingImage)
            #     cv2.imwrite(os.path.join('segmented', 'by_element',  str(r) + '_' + str(ele) + '.jpg'), maskingImage)

            num += 1

        return label_color

    def segmentationStreetview(self, streetviewName, network, resized_folder, segmented_folder):
        print("os.path.join(resized_folder, streetviewName:", os.path.join(resized_folder, streetviewName))

        parser = argparse.ArgumentParser()
        parser.add_argument("model_dir", nargs='?', default=
        "C:/Users/yuriK/OneDrive/ドキュメント/#colour/code/Tensorflow-Segmentation-master/Tensorflow-Segmentation-master/save/C7,64,2C7,64,1M2C7,64,2C7,64,1M2C7,64,2C7,64,1M2/2018-08-09_035250"
        "", type=str, help="Path to directory storing checkpointed model.")# best performance2018-08-09_035250
        parser.add_argument("test_image", nargs='?', default=os.path.join(resized_folder, streetviewName), type=str, help="Path to image for which the segmentation should be performed.")
        parser.add_argument("--out", default="/tmp", type=str, help="Path to directory to store resulting image.")#if not list, define nargs as ?
        print(parser)

        args = parser.parse_args()


        test_image = args.test_image
        checkpoint = args.model_dir

        with tf.Session() as sess:

            saver = tf.train.Saver(tf.global_variables())#initially tf.all_variables()
            ckpt = tf.train.get_checkpoint_state(checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError('No model found in {}.'.format(checkpoint))

            image = cv2.imread(test_image, 1) # load grayscale* changed to color scale because n>0 = 1


            # remain image color before segmentation as rgb_image (converted bgr to rgb)
            bgr_image = image
            rgb_image = bgr_image[:, :, [2, 1, 0]]


            #Resize input image
            # もとの写真を正方形にcropする
            inh, inw, inc = image.shape
            print("inh, inw, inc: ", inh, inw, inc)
            if (inh != inw):
                print("image shape: inh != inw")
                if (inh <= inw):
                    image = image[20:20+inh, 0:inh] #y1:y2, x1:x2
                else:
                    image = image[20:20+inw, 0:inw]
                image = cv2.resize(image, (network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
                cv2.imwrite(os.path.join('input_resized.jpg'), image)

            image = np.array(image)
            image = np.multiply(image, 1.0 / 255)

            segmentation = sess.run(network.segmentation_result, feed_dict={
                network.inputs: np.reshape(image, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS])})
            print("segmentation_len: ", len(segmentation))
            segNormalized = segmentation[0]
            segmented_image = np.dot(segmentation[0], 255)

            label_image = []

            # segmented_image = segmented_image[:, :, [2, 1, 0]]
            cv2.imwrite(os.path.join(segmented_folder, 'segmented_before_' + streetviewName[:-4] + '.jpg'), segmented_image)

            for x, clm in enumerate(segmented_image):
                label_image.append([])
                for y, pxl in enumerate(clm):
                    # # assign method
                    # for z, col in enumerate(pxl):
                    #     # print(pxl, "pxl")
                    #     if (0 <= col <= 63.75):
                    #         # z = 0
                    #         segmented_image[x][y][z] = 0
                    #     elif (63.75 < col <= 127.5):
                    #         # z = 85
                    #         segmented_image[x][y][z] = 85
                    #     elif (127.5 < col <= 191.25):
                    #         # z = 170
                    #         segmented_image[x][y][z] = 170
                    #     elif (191.25 < col <= 255):
                    #         # z = 255
                    #         segmented_image[x][y][z] = 255
                    #     else:
                    #         pass

                    # distance method

                    distances = []
                    for z,label_value in enumerate(clustering.label_array):


                        # dist = Cluster.euclidean_normal(pxl, x, y, z) # change this part for better segmentation labels

                        #eucldean lowcost
                        dist = Cluster.euclidean_lowcost(pxl, x, y, z)  # change this part for better segmentation labels

                        # #cie2000
                        # lab0 = Cluster.rgb2lab(pxl)
                        # lab1 = Cluster.rgb2lab(clustering.label_centroids[z])
                        # print("pxl <----> clustering.label_centroids[z]: ", pxl, "<--->", clustering.label_centroids[z])
                        # print("Lab0*: ", lab0, "/ Lab1*: ", lab1)
                        # dist = Cluster.cie2000(lab0, lab1)
                        # print("dist: ", dist)

                        distances.append(dist)
                    # labelName = elementsNames.index(min(distances))
                    # print("distances:", distances)
                    # print("min index in distances :", distances.index(min(distances)))

                    labelValue = clustering.label_array[distances.index(min(distances))]

                    # print("labelValue: ", labelValue)
                    segmented_image[x][y] = labelValue

                    labels = []
                    difference = []

                    for k, v in clustering.label_dict.items():
                        labels.append(k)
                        diff = 1 / (1 + np.sqrt(np.square(abs(segmented_image[x][y][0]-v[0])) + np.square(abs(segmented_image[x][y][1]-v[1])) + np.square(abs(segmented_image[x][y][2]-v[2]))))
                        #diff = abs(pxl[0]-v[0]) + abs(pxl[1]-v[1]) + abs(pxl[2]-v[2])
                        difference.append(diff)
                    difference = np.array(difference)
                    # min = difference.argmin()
                    max = difference.argmax()

                    label_image[x].append(labels[max])
                    segmented_image[x][y] = clustering.label_dict[labels[max]]

            # print("##############################################################################")
            # print(label_image[0], "label_max[0]")
            # print("##############################################################################")

            cv2.imwrite(os.path.join(segmented_folder, 'segmented_' + streetviewName[:-4] + '.jpg'), segmented_image)

            return label_image, rgb_image

            # segmented_image = cv2.COLOR_RGB2BGR(segmented_image)

if __name__ == '__main__':

    print("##############################################################################")



    # network = convolutional_autoencoder.Network(layers)

    test = Test()