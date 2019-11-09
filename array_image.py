import os
import cv2
import datetime
import numpy as np
import matplotlib
import shutil
import math
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

import datetime
import io

class ArrayImages:
    def __init__(self, folder):
        # folder = r"C:\Users\yuriK\OneDrive\ドキュメント\#colour\code\Tensorflow-Segmentation-master\Tensorflow-Segmentation-master\DlByArea\Amsteldam_01\10_palette"

        # C:\Users\yuriK\OneDrive\ドキュメント\  # colour\code\Tensorflow-Segmentation-master\Tensorflow-Segmentation-master\DlByArea\Paris_Av. des Champs-Élysées\10_palette
        self.arrayImages(folder)

    def arrayImages (self, folder):



        # folder = os.path.join(folder, folder_test)
        # print("folder: ", folder)

        folder_streetview = os.path.join(folder, 'streetview')
        folder_segmented = os.path.join(folder, 'segmented')
        folder_DLimgs = os.path.join(folder, 'DLimgs')

        files_streetview = os.listdir(folder_streetview)
        files_segmented = os.listdir(folder_segmented)
        files_DLimgs = os.listdir(folder_DLimgs)
        # print("files_segmented:", files_segmented)

        array_dir = os.path.join(folder, "array")

        if os.path.exists(array_dir):
            shutil.rmtree(array_dir)
            print("array_dir DELETED")
        os.makedirs(array_dir)

        # print("(os.path.join(folder_streetview, files_streetview[0])):", (os.path.join(folder_streetview, files_streetview[0])))
        # print("files_streetview[0]: ", files_streetview[0])
        imagesize = cv2.imread((os.path.join(folder_streetview, files_streetview[0])), 1)
        print("image: ", os.path.join(folder_streetview, files_streetview[0]))
        # print("imagesize: ", imagesize)
        height, width = imagesize.shape[:2]
        dammy_canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        imagesize = cv2.imread((os.path.join(folder_DLimgs, files_DLimgs[0])), 1)
        # print("imagesize: ", imagesize)
        height_DLimgs, width_DLimgs = imagesize.shape[:2]
        dammy_canvas_DLimgs = np.full((height_DLimgs, width_DLimgs, 3), 255, dtype=np.uint8)


        # streetview cropped ####################################################################
        streetview_array = os.path.join(array_dir, "streetview")
        # print("streetview_array", streetview_array)
        os.makedirs(streetview_array)
        # vertical_images = []
        for i in range(math.ceil(len(files_streetview)/12)):
            if (i+1)*12 < (len(files_streetview)):
                horizon_images = []
                for j in range(12):
                    # print("streetview i:", i, "j:", j)
                    image_read = cv2.imread(os.path.join(folder_streetview, files_streetview[i*12 + j]), 1)
                    horizon_images.append(image_read)
                imgs_h = cv2.hconcat(horizon_images)  # no palettes
                # vertical_images.append((imgs_h))
                cv2.imwrite(os.path.join(streetview_array, "horizon_streetview" + str(i) + ".jpg"), imgs_h)
            else:
                horizon_images = []
                for j in range(12):
                    if j < len(files_streetview)-12*i:
                        # print("len(files_streetview)-12*i: ", len(files_streetview)-12*i)
                        # print("streetview i:", i, "j:", j)
                        print("os.path.join(folder_streetview, files_streetview[i*12 + j]): ", os.path.join(folder_streetview, files_streetview[i*12 + j]))
                        image_read = cv2.imread(os.path.join(folder_streetview, files_streetview[i*12 + j]), 1)
                        horizon_images.append(image_read)
                    else:
                        # print("streetview dammy i:", i, "j:", j)
                        horizon_images.append(dammy_canvas)

                imgs_h = cv2.hconcat(horizon_images)  # no palettes
                # vertical_images.append((imgs_h))
                # print("os.path.join(array_dir, horizon_streetview" + str(i) + ".jpg:", os.path.join(array_dir, "horizon_streetview" + str(i) + ".jpg"))
                cv2.imwrite(os.path.join(streetview_array, "horizon_streetview" + str(i) + ".jpg"), imgs_h)

        vertical_images = []
        arrays = os.listdir(streetview_array)
        for i in range(len(arrays)):
            hor = cv2.imread(os.path.join(streetview_array, arrays[i]), 1)
            vertical_images.append((hor))
        imgs_v = cv2.vconcat(vertical_images)
        # print("img_v: ", imgs_v)
        cv2.imwrite(os.path.join(array_dir, "vertical_streetview.jpg"), imgs_v)










        # streetview cropped ####################################################################
        DLimgs_array = os.path.join(array_dir, "DLimgs")
        print("DLimgs_array: ", DLimgs_array)
        os.makedirs(DLimgs_array)
        # vertical_images = []
        for i in range(math.ceil(len(files_streetview) / 12)):
            if (i + 1) * 12 < (len(files_streetview)):
                horizon_images = []
                for j in range(12):
                    # print("streetview i:", i, "j:", j)
                    image_read = cv2.imread(os.path.join(folder_DLimgs, files_DLimgs[i * 12 + j]), 1)
                    horizon_images.append(image_read)
                imgs_h = cv2.hconcat(horizon_images)  # no palettes
                # vertical_images.append((imgs_h))
                cv2.imwrite(os.path.join(DLimgs_array, "horizon_DLimgs" + str(i) + ".jpg"), imgs_h)
            else:
                horizon_images = []
                for j in range(12):
                    if j < len(files_streetview) - 12 * i:
                        # print("len(files_streetview)-12*i: ", len(files_streetview) - 12 * i)
                        # print("streetview i:", i, "j:", j
                        print("i * 12 + j: ", i * 12 + j)
                        image_read = cv2.imread(os.path.join(folder_DLimgs, files_DLimgs[i * 12 + j]), 1)
                        horizon_images.append(image_read)
                    else:
                        # print("streetview dammy i:", i, "j:", j)
                        horizon_images.append(dammy_canvas_DLimgs)
                imgs_h = cv2.hconcat(horizon_images)  # no palettes
                # vertical_images.append((imgs_h))
                # print("os.path.join(array_dir, horizon_DLimgs" + str(i) + ".jpg:",                      os.path.join(array_dir, "horizon_DLimgs" + str(i) + ".jpg"))
                cv2.imwrite(os.path.join(DLimgs_array, "horizon_DLimgs" + str(i) + ".jpg"), imgs_h)

        vertical_images = []
        arrays = os.listdir(DLimgs_array)
        for i in range(len(arrays)):
            hor = cv2.imread(os.path.join(DLimgs_array, arrays[i]), 1)
            vertical_images.append((hor))
        imgs_v = cv2.vconcat(vertical_images)
        # print("img_v: ", imgs_v)
        cv2.imwrite(os.path.join(array_dir, "vertical_sDLimgs.jpg"), imgs_v)










        # array segmented view
        rsv = []
        before = []
        seg2 = [before, rsv]
        for v in range(len(files_segmented)):
            # print("files_segmented[v][20:]: ", files_segmented[v][:20], "files_segmented[v][13:]: ",
            #           files_segmented[v][:13])
            if files_segmented[v][:20] == "segmented_before_rsv":
                rsv.append(files_segmented[v])
            elif files_segmented[v][:13] == "segmented_rsv":
                before.append(files_segmented[v])
            else:
                # print("files_segmented[v][20:]: ", files_segmented[v][:20], "files_segmented[v][13:]: ",
                #       files_segmented[v][:13])
                pass

        for s, seg in enumerate(seg2):
            key = s
            segmented_array = os.path.join(array_dir, "segmented"+str(key))
            os.makedirs(segmented_array)
            print("segmented_array: ", segmented_array)
            for i in range(math.ceil(len(seg) / 12)):
                if (i + 1) * 12 < (len(seg)):
                    horizon_images = []
                    for j in range(12):
                        # print("segmented", str(key), "i:", i, "j:", j)
                        image_read = cv2.imread(os.path.join(folder_segmented, seg[i*12 + j]), 1)
                        horizon_images.append(image_read)
                    # print("horizon_images: ", horizon_images)
                    imgs_h = cv2.hconcat(horizon_images)  # no palettes
                    # vertical_images.append((imgs_h))
                    cv2.imwrite(os.path.join(segmented_array, "horizon_segmented" + str(i) + ".jpg"), imgs_h)
                else:
                    horizon_images = []
                    for j in range(12):
                        # print("segmented", str(key), "i:", i, "j:", j)
                        if j < len(seg) - 12 * i:
                            image_read = cv2.imread(os.path.join(folder_segmented, seg[i*12 + j]), 1)
                            horizon_images.append(image_read)
                        else:
                            horizon_images.append(dammy_canvas)

                    imgs_h = cv2.hconcat(horizon_images)  # no palettes
                    # vertical_images.append((imgs_h))
                    cv2.imwrite(os.path.join(segmented_array, "horizon_segmented" + str(i) + ".jpg"), imgs_h)

            vertical_images = []
            arrays = os.listdir(segmented_array)
            for i in range(len(arrays)):
                hor = cv2.imread(os.path.join(segmented_array, arrays[i]), 1)
                vertical_images.append((hor))
            imgs_v = cv2.vconcat(vertical_images)
            # print("vertical_images: ", vertical_images)
            # print("img_v: ", imgs_v)
            cv2.imwrite(os.path.join(array_dir, "vertical_segmented_" + str(key) + ".jpg"), imgs_v)

class ArrayPalette:
    def __init__(self, folder):


        self.arraypalettes(folder)

    def arraypalette (self, folder):

        ids = os.listdir(folder)
        margin = 10
        name = 220
        name_space = int(name-int(margin/2))

        paletteH = int(50 * 1 + margin *1)
        paletteW = int(50 * 10 + name)

        for i,id in enumerate(ids):
            id_path = os.path.join(folder, id)
            if os.path.isdir(id_path):
                palette_path = os.path.join(id_path, 'palette')
                if os.path.exists(palette_path):
                    palettes =  os.listdir(palette_path)
                    all_palette = []
                    elementsNames = ["background", "facade", "window", "door", "cornice", "sill", "balcony",
                                     "blind", "deco", "molding", "pillar", "shop"]
                    for p,pale in enumerate(elementsNames):
                        pale_path = os.path.join(palette_path, pale+".jpg")
                        print("proceeding: ", pale_path)
                        palette_canvas = np.full((paletteH, paletteW, 3), 255, dtype=np.uint8)
                        cv2.putText(palette_canvas, pale, (int(margin/2), 20+int(margin/2) + margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                        if os.path.exists(pale_path):
                            palette = cv2.imread(pale_path, 1)
                            p_h, p_w = palette.shape[:2]
                            c_h, c_w = palette_canvas.shape[:2]
                            print("p_w: ", p_w)
                            if p_w>500:
                                p_w = 500
                                palette = palette[0:p_h, 0:p_w]
                            else:
                                palette = palette
                            palette_canvas[int(margin/2):int(margin/2)+p_h, name
                                                  :name+p_w] = palette
                            all_palette.append(palette_canvas)
                        else:
                            all_palette.append(palette_canvas)
                    rect_palette = cv2.vconcat(all_palette)
                    hh, ww = rect_palette.shape[:2]
                    print("hh: ", hh, "  ww: ", ww)
                    margin_canvas = np.full((1070, 1070, 3), 255, dtype=np.uint8)
                    resize_palette = cv2.resize(rect_palette, (1030, 1030))
                    margin_canvas[20:1050, 20:1050] = resize_palette
                    print("all palette created in folder: ", palette_path)

                    cv2.imwrite(os.path.join(palette_path, "all_palette.jpg"), margin_canvas)
                else:
                    print(palette_path, ":does not exist")
            else:
                print(id_path, ":not a folder")

    #for clastering num
    def arraypalettes (self, folder):

        ids = os.listdir(folder)
        margin = 10
        name = 220
        name_space = int(name-int(margin/2))

        paletteH = int(50 * 1 + margin *1)
        paletteW = int(50 * 10 + name)

        for i,id in enumerate(ids):
            id_path = os.path.join(folder, id)
            palettes =  os.listdir(id_path)
            all_palette = []
            elementsNames = ["background", "facade", "window", "door", "cornice", "sill", "balcony",
                             "blind", "deco", "molding", "pillar", "shop"]
            for p,pale in enumerate(elementsNames):
                pale_path = os.path.join(id_path, pale+".jpg")
                print("proceeding: ", pale_path)
                palette_canvas = np.full((paletteH, paletteW, 3), 255, dtype=np.uint8)
                cv2.putText(palette_canvas, pale, (int(margin/2), 20+int(margin/2) + margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1,
                            cv2.LINE_AA)
                if os.path.exists(pale_path):
                    palette = cv2.imread(pale_path, 1)
                    p_h, p_w = palette.shape[:2]
                    c_h, c_w = palette_canvas.shape[:2]
                    print("p_w: ", p_w)
                    if p_w>500:
                        p_w = 500
                        palette = palette[0:p_h, 0:p_w]
                    else:
                        palette = palette
                    palette_canvas[int(margin/2):int(margin/2)+p_h, name
                                          :name+p_w] = palette
                    all_palette.append(palette_canvas)
                else:
                    all_palette.append(palette_canvas)
            rect_palette = cv2.vconcat(all_palette)
            hh, ww = rect_palette.shape[:2]
            print("hh: ", hh, "  ww: ", ww)
            margin_canvas = np.full((1070, 1070, 3), 255, dtype=np.uint8)
            resize_palette = cv2.resize(rect_palette, (1030, 1030))
            margin_canvas[20:1050, 20:1050] = resize_palette
            print("all palette created in folder: ", id_path)

            cv2.imwrite(os.path.join(id_path, "all_palette.jpg"), margin_canvas)



# array images for theses, just DLimgs
def array(folders_path):

    folders = os.listdir(folders_path)

    for folder in folders:


        folder_path = os.path.join(folders_path, folder)


        files = os.listdir(folder_path)



        imagesize = cv2.imread(img_path, 1)
        print("imagesize: ", imagesize)

        height, width = imagesize.shape[:2]
        # height = 188
        # width = 128
        dammy_canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        array_dir = os.path.join(folders_path, folder, "array")

        if os.path.exists(array_dir):
            shutil.rmtree(array_dir)
            print("array_dir DELETED")
        os.makedirs(array_dir)
        horizontal_array = os.path.join(array_dir, "horizontal")
        os.makedirs(horizontal_array)

        if "array" in files:
            print("files: ", files)
            files.remove('array')
            print("files: ", files)

        # vertical_images = []
        for i in range(math.ceil(len(files) / 12)):
            if (i + 1) * 12 < (len(files)):
                horizon_images = []
                for j in range(12):
                    image_read = cv2.imread(os.path.join(folder_path, files[i * 12 + j]), 1)
                    horizon_images.append(image_read)
                imgs_h = cv2.hconcat(horizon_images)  # no palettes
                cv2.imwrite(os.path.join(horizontal_array, "horizontal" + str(i) + ".jpg"), imgs_h)
            else:
                horizon_images = []
                for j in range(12):
                    if j < len(files) - 12 * i:
                        image_read = cv2.imread(os.path.join(folder_path, files[i * 12 + j]), 1)
                        horizon_images.append(image_read)
                    else:
                        horizon_images.append(dammy_canvas)

                imgs_h = cv2.hconcat(horizon_images)  # no palettes
                # vertical_images.append((imgs_h))
                # print("os.path.join(array_dir, horizon_streetview" + str(i) + ".jpg:", os.path.join(array_dir, "horizon_streetview" + str(i) + ".jpg"))
                cv2.imwrite(os.path.join(horizontal_array, "horizon_streetview" + str(i) + ".jpg"), imgs_h)

        vertical_images = []
        arrays = os.listdir(horizontal_array)
        for i in range(len(arrays)):
            hor = cv2.imread(os.path.join(horizontal_array, arrays[i]), 1)
            vertical_images.append((hor))
        imgs_v = cv2.vconcat(vertical_images)
        # print("img_v: ", imgs_v)
        cv2.imwrite(os.path.join(array_dir, str(folder)+"array.jpg"), imgs_v)


if __name__ == '__main__':
    start = datetime.datetime.today()
    print('start time:', start)
    print("##############################################################################")
    domain = r"DlByArea\CAADRIA"
    folder_tests = os.listdir(domain)

    # for folder in folder_tests:
    #     print("folder: ", folder)
    #     folder_path = os.path.join(domain, folder)
    #     array = ArrayImages(folder_path)

    # pale = ArrayPalette(domain)
    arr = ArrayImages(domain)

    # domain = r"DlByArea\00_test_condition\000"
    # array(domain)

    print("##############################################################################")
    end = datetime.datetime.today()
    print('end time:', end)