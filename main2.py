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
import array_image
from main import Main
import datetime
import shutil
from mutagen.mp3 import MP3 as mp3
import pygame
import time
import random


class Main2:
    def __init__(self):
        street_folder = r"DlByArea\000_test"
        self.move_imgs(street_folder)
        self.infer_by_unit(street_folder)

    def move_imgs(self, folder):
        units = os.listdir(folder)
        for i,unit in enumerate(units):
            unit_path = os.path.join(folder, unit)
            if os.path.isdir(unit_path):
                unit_list = os.listdir(unit_path)

                DLimgs_path = os.path.join(unit_path, "DLimgs")
                streetview_path = os.path.join(unit_path, "streetview")
                segmented_path = os.path.join(unit_path, "segmented")
                palette_path = os.path.join(unit_path, "palette")

                if os.path.exists(DLimgs_path):
                    shutil.rmtree(DLimgs_path)
                    shutil.rmtree(streetview_path)
                    shutil.rmtree(segmented_path)
                    shutil.rmtree(palette_path)
                    print("array_dir DELETED")
                else:
                    pass
                os.makedirs(DLimgs_path)
                os.makedirs(streetview_path)
                os.makedirs(segmented_path)
                os.makedirs(palette_path)
                unit_list = os.listdir(unit_path)
                # print("DLimgs_path:", DLimgs_path)
                # print("unit_list:", unit_list)
                DLimgs_len = 0
                for o, obj in enumerate(unit_list):
                    # print("obj[:6]", obj[:6])
                    if obj[:6] == "DLimgs" and obj !="DLimgs":
                        obj_path = os.path.join(unit_path, obj)
                        img_list = os.listdir(obj_path)
                        for i,img in enumerate(img_list):
                            original_img_path = os.path.join(obj_path, img)
                            copied_img_path = os.path.join(DLimgs_path, "streetview" + str(DLimgs_len) + ".jpg")
                            shutil.copy(original_img_path, copied_img_path)
                            DLimgs_len += 1
                print("copy finished")
            else:
                print("this is not a directry: ", unit_path)


    def infer_by_unit(self, folder):
        units = os.listdir(folder)
        for i,unit in enumerate(units):
            unit_path = os.path.join(folder, unit)
            if os.path.isdir(unit_path):
                unit_list = os.listdir(unit_path)

                DLimgs_path = os.path.join(unit_path, "DLimgs")
                streetview_path = os.path.join(unit_path, "streetview")
                segmented_path = os.path.join(unit_path, "segmented")
                palette_path = os.path.join(unit_path, "palette")

                if os.path.exists(DLimgs_path):
                    print("Do the infer thing No. ", i, " - ", DLimgs_path, "!!!!!!!!!!!!!!!!!!!!")
                    streetViewTest = Main(folder=DLimgs_path, resized_folder=streetview_path, segmented_folder=segmented_path, palette_folder=palette_path, load_DLimages=True, load_Cluster=True)
                    finish = datetime.datetime.today()
                    print('proceeding ', DLimgs_path, ': ', finish)
                    music_play(filename='alarm', loop=5, repeat=1)
                else:
                    print("DLimgs folder does not exists")
            else:
                print("this is not a directry: ", unit_path)


def music_play(filename='alarm', loop=3, repeat=3):

    for i in range(repeat):
        alarm_file = os.path.join(filename, random.choice(os.listdir(filename)))  # 再生したいmp3ファイル
        pygame.mixer.init()
        pygame.mixer.music.load(alarm_file)  # 音源を読み込み
        mp3_length = mp3(alarm_file).info.length  # 音源の長さ取得
        pygame.mixer.music.play(loop)  # 再生開始。1の部分を変えるとn回再生(その場合は次の行の秒数も×nすること)
        time.sleep(mp3_length*loop*1.0 + 0.05)  # 再生開始後、音源の長さだけ待つ(0.25待つのは誤差解消)
        pygame.mixer.music.stop()  # 音源の長さ待ったら再生停止

if __name__ == '__main__':

    start = datetime.datetime.today()
    print('start time:', start)

    print("##############################################################################")
    # print("Network.IMAGE_HEIGHT, Network.IMAGE_WIDTH:", Network.IMAGE_HEIGHT, Network.IMAGE_WIDTH)
    # print("##############################################################################")

    # print(" = self.file_paths_to_DLimages(folder, os.listdir(os.path.join(folder, 'DLimgs'))", os.listdir(os.path.join("DLimgs"))    )


    streetsTest = Main2()
    domain = r"DlByArea\000_test"
    folder_tests = os.listdir(domain)

    for folder in folder_tests:
        print("folder: ", folder)
        folder_path = os.path.join(domain, folder)
        array = array_image.ArrayImages(folder_path)

    pale = array_image.ArrayPalette(domain)

    music_play()

    print("##############################################################################")
    end = datetime.datetime.today()
    print('end time:', end)

