import os
import numpy as np
import cv2
import convolutional_autoencoder
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import math
from convolutional_autoencoder import Network
import infer
import icon
import clustering
from clustering import Cluster
import matplotlib.pyplot as plt
import pickle
import datetime
import shutil
from mutagen.mp3 import MP3 as mp3
import pygame
import time
import random



class Main:
    def __init__(self, folder, resized_folder, segmented_folder, palette_folder, load_DLimages, load_Cluster): # load DLimages
        #def __init__(self, folder, resized_folder,  segmented_folder, palette_folder, load_DLimages, load_Cluster)
        #self.folder = folder
        # self.resized_folder = resized_folder
        # self.segmented_folder = segmented_folder
        # self.palette_folder = palette_folder

        self.folder = folder
        self.resized_folder = resized_folder
        self.segmented_folder = segmented_folder
        self.palette_folder = palette_folder
        self.load_DLimages = load_DLimages

        label_color = self.load(self.load_DLimages)

        # 検証、カラーパレットの初期値数１～２０まで
        # print("remember.label_color:", label_color)
        # for clusterNum in range(1, 20):
        #     cluster_folder = os.path.join(palette_folder, "clusterNum" + str(clusterNum))
        #     if os.path.exists(cluster_folder):
        #         shutil.rmtree(cluster_folder)
        #         print("cluster_folder:", clusterNum, " DELETED")
        #     os.makedirs(cluster_folder)
        #     self.elementsColor, self.usedElementsNames = self.callCluster(cluster_folder,  label_color, clusterNum)
        #     self.draw = icon.drawIcon(self.elementsColor, self.usedElementsNames, cluster_folder)

        # normal version of color palette
        self.elementsColor, self.usedElementsNames, self.elements_len = self.callCluster(label_color, palette_folder, clusterNum=10) # use this

        # self.draw_line = icon.pattern_random(self.elementsColor, self.usedElementsNames, palette_folder, self.elements_len)
        # self.draw_line = icon.pattern_random_hatch(self.elementsColor, self.usedElementsNames, palette_folder, self.elements_len)

        # # hue segmentation version of color palette
        # self.elementsColor, self.usedElementsNames= self.callClusterHSV(palette_folder, label_color, clusterNum=10)
        # self.draw_line = icon.pattern_random(self.elementsColor, self.usedElementsNames, palette_folder, elements_len=0)
        self.draw_line = icon.pattern_random_hatch(self.elementsColor, self.usedElementsNames, palette_folder, elements_len=0) # use this

    def load(self, load_DLimages):
        if load_DLimages == True:
            print ("load_DLimages = True")
            remember = infer.Test(self.folder, self.resized_folder, self.segmented_folder)
            with open('streetviewimage.', mode='wb') as save:
                pickle.dump(remember, save)
        elif load_DLimages ==False:
            print ("load_DLimages = False")
            with open('streetviewimage.', mode='rb') as save:
                remember = pickle.load(save)
        else:
            pass
        return remember.label_color

    @classmethod
    def callCluster(self, label_color, palette_folder, clusterNum):
        elementsColor = []
        usedElementsNames = []
        elements_len = []
        for i in range(len(label_color)):
            print("len_label_color:::", len(label_color[i]))
            if(len(label_color[i])>0):
                    sortCentroids, sortClusters_len = Cluster.makeClusters(label_color[i], clustering.elementsNames[i], palette_folder, clusterNum)
                    # sortCentroids, sortClusters_len = Cluster.makeClusters_HSV(label_color[i], clustering.elementsNames[i], palette_folder, clusterNum)
                    # sortCentroids, sortClusters_len = Cluster.makeClusters_HSVcentroids(label_color[i], clustering.elementsNames[i], palette_folder, clusterNum)

                    usedElementsNames.append(clustering.elementsNames[i])
                    elementsColor.append(sortCentroids)
            else:
                elementsColor.append([[-1, -1, -1], [-1, -1, -1]])
        print("usedElementsNames", usedElementsNames)
        return elementsColor, usedElementsNames, elements_len

    @classmethod
    def callClusterHSV(self, label_color, palette_folder, clusterNum):
        elementsColor = []
        usedElementsNames = []
        elements_len = []
        for i in range(len(label_color)):
            print("len_label_color:::", len(label_color[i]))
            if(len(label_color[i])>0):
                    sortCentroids= Cluster.makeClusters_HSV(label_color[i], clustering.elementsNames[i], palette_folder, clusterNum)

                    usedElementsNames.append(clustering.elementsNames[i])
                    elementsColor.append(sortCentroids)
            else:
                elementsColor.append([[-1, -1, -1], [-1, -1, -1]])
        print("usedElementsNames", usedElementsNames)
        return elementsColor, usedElementsNames

def music_play(filename='alarm', loop=3):

    for i in range(5):
        alarm_file = os.path.join(filename, random.choice(os.listdir(filename)))  # 再生したいmp3ファイル
        pygame.mixer.init()
        pygame.mixer.music.load(alarm_file)  # 音源を読み込み
        mp3_length = mp3(alarm_file).info.length  # 音源の長さ取得
        pygame.mixer.music.play(loop)  # 再生開始。1の部分を変えるとn回再生(その場合は次の行の秒数も×nすること)
        time.sleep(mp3_length*loop + 0.05)  # 再生開始後、音源の長さだけ待つ(0.25待つのは誤差解消)
        pygame.mixer.music.stop()  # 音源の長さ待ったら再生停止

if __name__ == '__main__':

    start = datetime.datetime.today()
    print('start time:', start)

    print("##############################################################################")
    # print("Network.IMAGE_HEIGHT, Network.IMAGE_WIDTH:", Network.IMAGE_HEIGHT, Network.IMAGE_WIDTH)
    # print("##############################################################################")

    # print(" = self.file_paths_to_DLimages(folder, os.listdir(os.path.join(folder, 'DLimgs'))", os.listdir(os.path.join("DLimgs"))    )


    streetViewTest = Main(folder="DLimgs", resized_folder="streetview",  segmented_folder="segmented", palette_folder="palette", load_DLimages=True, load_Cluster=True)
    # music_play()

    print("##############################################################################")
    end = datetime.datetime.today()
    print('end time:', end)


