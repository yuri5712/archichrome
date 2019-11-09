import os
import numpy as np
import cv2
import math
import convolutional_autoencoder
import icon
import matplotlib.pyplot as plt
import shutil

# label of CMP dictionaly BGR
label_dict = dict([('no_label', [0, 0, 0]),
                   ('background', [170, 0, 0]),
                   ('facade', [255, 0, 0]),
                   ('window', [255, 85, 0]),
                   ('door', [255, 170, 0]),
                   ('cornice', [255, 255, 0]),
                   ('sill', [170, 255, 85]),
                   ('balcony', [85, 255, 170]),
                   ('blind', [0, 255, 255]),
                   ('deco', [0, 170, 255]),
                   ('molding', [0, 85, 255]),
                   ('pillar', [0, 0, 255]),
                   ('shop', [0, 0, 170])])
label_array = [[0, 0, 0],
               [170, 0, 0],
               [255, 0, 0],
               [255, 85, 0],
               [255, 170, 0],
               [255, 255, 0],
               [170, 255, 85],
               [85, 255, 170],
               [0, 255, 255],
               [0, 170, 255],
               [0, 85, 255],
               [0, 0, 255],
               [0, 0, 170]]
label_centroids = [[31.875, 31.875, 31.875],
                   [159.375, 31.875, 31.875],
                   [223.125, 31.875, 31.875],
                   [223.125, 95.625, 31.875],
                   [223.125, 159.375, 31.875],
                   [223.125, 223.125, 31.875],
                   [159.375, 223.125, 95.625],
                   [95.625, 223.125, 159.375],
                   [31.875, 223.125, 223.125],
                   [31.875, 159.375, 223.125],
                   [31.875, 95.625, 223.125],
                   [31.875, 31.875, 223.125],
                   [31.875, 31.875, 159.375]]
label_normal = [[0, 0, 0],
                [0.666, 0, 0],
                [1, 0, 0],
                [1, 0.333, 0],
                [1, 0.666, 0],
                [1, 1, 0],
                [0.666, 1, 0.333],
                [0.333, 1, 0.666],
                [0, 1, 1],
                [0, 0.666, 1],
                [0, 0.333, 1],
                [0, 0, 1],
                [0, 0, 0.666]]
label_ratio = [[0, 0, 0],
               [1.0, 0, 0],
               [1.0, 0, 0],
               [0.75, 0.25, 0],
               [0.6, 0.3, 0],
               [0.5, 0.5, 0],
               [0.333, 0.5, 0.166],
               [0.166, 0.5, 0.333],
               [0, 0.5, 0.5],
               [0, 0.4, 0.6],
               [0, 0.25, 0.75],
               [0, 0, 1.0],
               [0, 0, 1.0]]
label_dict_recolor = dict([('no_label', [0, 0, 0]),
                   ('background', [85, 0, 0]),
                   ('facade', [170, 0, 0]),
                   ('window', [255, 85, 0]),
                   ('door', [255, 170, 0]),
                   ('cornice', [255, 255, 85]),
                   ('sill', [255, 255, 170]),
                   ('balcony', [255, 85, 255]),
                   ('blind', [170, 0, 255]),
                   ('deco', [85, 0, 255]),
                   ('molding', [0, 85, 255]),
                   ('pillar', [0, 0, 170]),
                   ('shop', [0, 0, 85])])
label_recolor = [[0, 0, 0],
               [85, 0, 0],
               [170, 0, 0],
               [255, 85, 0],
               [255, 170, 0],
               [255, 255, 85],
               [255, 255, 170],
               [255, 170, 255],
               [255, 85, 255],
               [170, 0, 255],
               [85, 0, 255],
               [0, 0, 170],
               [0, 0, 85]]


elementsNames = ["no_label", "background", "facade", "window", "door", "cornice", "sill", "balcony", "blind",
                 "deco", "molding", "pillar", "shop"]

# label of CMP dictionaly RGB
label_dict_RGB = dict([('no_label', [0, 0, 0]), ('background', [0, 0, 170]), ('facade', [0, 0, 255]), ('window', [0, 85, 255]), ('door', [0, 170, 255]),
          ('cornice', [0, 255, 255]), ('sill', [85, 255, 170]), ('balcony', [170, 255, 85]),
          ('blind', [255, 255, 0]), ('deco', [255, 170, 0]), ('molding', [255, 85, 0]), ('pillar', [255, 0, 0]),
          ('shop', [170, 0, 0])])

RGB2XYZ_D65 = (
    0.412453, 0.357580, 0.180423,
    0.212671, 0.715160, 0.072169,
    0.019334, 0.119193, 0.950227
)

D65 = ( 0.950456, 1., 1.088754 )

_coff = (RGB2XYZ_D65[0]*(1.0/D65[0]),RGB2XYZ_D65[1]*(1.0/D65[0]),RGB2XYZ_D65[2]*(1.0/D65[0]),
     RGB2XYZ_D65[3]*(1.0/D65[1]),RGB2XYZ_D65[4]*(1.0/D65[1]),RGB2XYZ_D65[5]*(1.0/D65[1]),
     RGB2XYZ_D65[6]*(1.0/D65[2]),RGB2XYZ_D65[7]*(1.0/D65[2]),RGB2XYZ_D65[8]*(1.0/D65[2]),)


class Cluster:
    def __init__(self, color_list, centroids):
        # self.clustering = makeClusters(color, name)
        print ("clustering")

    # sort self.color_list by 12 clusters ##############################################################################
    @classmethod
    def makeClusters(self, color_list, name, palette_folder, clusterNum):
        minimum = np.array(color_list).min()
        maximum = np.array(color_list).max()
        # minimum = 0
        # maximum = 255
        print ("min:", minimum, "max:", maximum)

        # make a plot of color for this building element
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(color_list)

        #number of color for a palette for this building element
        clusterNum = clusterNum #12-->64
        # centroids = []
        # for rc in range(4):
        #     for gc in range(4):
        #         for bc in range(4):
        #             centroids.append([85*bc, 85*gc, 85*rc])
        centroids = np.random.randint(minimum-minimum*0.1, maximum+maximum*0.1, (clusterNum, 3))
        print ("centroids:", centroids)
        distortion = 0

        # make spaces for each clusters
        clusters = [[] for i in range(clusterNum)]


        # repeat iteration until distortion = distortion*0.005% or after 50 try
        for iterationNum in range(50):
            print("############################################")
            print("element name:", name)
            print("iterationNum:", iterationNum)
            # clusters = []
            distortion_new = 0

            #calc dis from centroids to each color coordinate
            for color in color_list:
                dists = []
                for centroid in centroids:
                    dist = np.sqrt(np.square(abs(color[0] - centroid[0])) + np.square(abs(color[1] - centroid[1])) + np.square(abs(color[2] - centroid[2])))
                    dists.append(dist)
                # color appended to min distant cluster
                clusterNo = dists.index(min(dists))
                # if(clusterNo.type == 'list'):
                #     clusters[clusterNo[0]].append(color)
                # else:
                clusters[clusterNo].append(color)
            print ("original cluster num:", len(clusters))
            # print("clusters:", clusters)
            # print("centroids:", centroids)

            # #delete empty array in clusters and centroids
            # tmp_clusters = []
            # tmp_centroids = []
            # for i,x in enumerate(clusters):
            #     if (x != []):
            #         tmp_clusters.append(x)
            #         tmp_centroids.append(centroids[i])
            # clusters = tmp_clusters
            # centroids = tmp_centroids
            # # clusters = [x for x in clusters if x]
            # print ("procedure cluster num:", len(clusters))


            for i in range(len(clusters)):
                # print("cluster", i, ":", len(clusters[i]))
                r = 0
                g = 0
                b = 0
                if (len(clusters[i])>0):
                    for col in clusters[i]:
                        r += col[0]
                        g += col[1]
                        b += col[2]
                    r = r/len(clusters[i])
                    g = g/len(clusters[i])
                    b = b/len(clusters[i])
                    newCentroid = [r, g, b]
                    # newCentroid = centroids[i]
                    # distortion_new.append(np.sqrt(np.square(abs(centroids[i][0] - newCentroid[0])) + np.square(abs(centroids[i][1] - newCentroid[1])) + np.square(abs(centroids[i][2] - newCentroid[2]))))
                    distortion_new += np.sqrt(np.square(abs(centroids[i][0] - newCentroid[0])) + np.square(abs(centroids[i][1] - newCentroid[1])) + np.square(abs(centroids[i][2] - newCentroid[2])))
                    centroids[i] = newCentroid
                elif(clusters[i] == []):
                    ##clusters = [x for x in clusters if x]
                    # np.delete(clusters, i, 0)
                    # np.delete(centroids, i, 0)
                    distortion_new += distortion_new/len(clusters)
                    # centroids[i] = centroids[i]
                else:
                    pass

            print("distortion:", distortion)
            print("distortion_new;", distortion_new)
            if (iterationNum > 0 and (abs(distortion - distortion_new)) < (distortion * 0.05)):
                break
            distortion = distortion_new

        # print ("centroids:", centroids)

        #delete empty clusters
        sortClusters = []
        for i in range(len(clusters)):
            if (clusters[i] !=  []):
                sortClusters.append(clusters[i])

        #sort clusters and centroids by length then draw rectangles
        # sortClusters = sortClusters.sort(key=len)
        sortClusters = sorted(sortClusters, key=len, reverse=True)
        # for i in range(len(sortClusters)):
        #     sortClusters[i] = list(sortClusters[i])
        centroids = list(centroids)
        sortCentroids = []

        clusters_len = []
        sortClusters_len = []
        for num in range(len(clusters)):
            # print("len(clusters[" + str(num) + "]:" + str(len(clusters[num])))
            clusters_len.append(len(clusters[num]))
        print("##############################################################################")
        for num in range(len(sortClusters)):
            # print("len(sortClusters[" + str(num) + "]:" + str(len(sortClusters[num])))
            sortClusters_len.append(len(sortClusters[num]))

        # print("sortClusters", sortClusters)
        print("len(sortClusters):", len(sortClusters))
        print("clusters_len:", clusters_len)
        print("sortClusters_len:", sortClusters_len)

        sortCentroids = icon.drawPalette(clusters, sortClusters, clusters_len, sortClusters_len, centroids, sortCentroids, name, palette_folder)

        return sortCentroids, sortClusters_len

    @classmethod
    def makeClusters_HSV(self, color_list, name, palette_folder, clusterNum):

        cluster_palette_folder = os.path.join(palette_folder, name)
        if os.path.exists(cluster_palette_folder):
            shutil.rmtree(cluster_palette_folder)
            print("cluster_folder:", name, " DELETED")
        os.makedirs(cluster_palette_folder)

        # create 10 hue class
        hue_array = [[] for hue_a in range(10)]
        hue_palette = []
        hue_palette_len = []

        for bgr_color in color_list:
            hsv_color = self.rgb2hsv(bgr_color)
            if(hsv_color[0] < 0):
                hsv_color[0] = 360 - hsv_color[0]
            elif(hsv_color[0] > 360):
                hsv_color[0] = hsv_color[0] % 360

            if (hsv_color[0] == 0 or hsv_color[0] == 360):
                hue_array[0].append(bgr_color)
            else:
                hue_num = int(hsv_color[0]//36)
                print("hue_num: ", hue_num)
                hue_array[hue_num].append(bgr_color)

        print("hue_array: ", hue_array)

        for hue,hue_color in enumerate(hue_array):
            if hue_color != []:
                # 初期値の範囲
                minimum = np.array(hue_color).min()
                maximum = np.array(hue_color).max()

                #number of color for a palette for this building element
                clusterNum = clusterNum #12-->64
                centroids = np.random.randint(minimum-minimum*0.1, maximum+maximum*0.1, (clusterNum, 3))
                print ("centroids:", centroids)
                distortion = 0
                clusters = []
                # make spaces for each clusters
                for i in range(clusterNum):
                    clusters.append([])

                # repeat iteration until distortion = distortion*0.005% or after 50 try
                for iterationNum in range(50):
                    print("############################################")
                    print("element name:", name)
                    print("iterationNum:", iterationNum)
                    # clusters = []
                    distortion_new = 0

                    #calc dis from centroids to each color coordinate
                    for color in hue_color:
                        dists = []
                        for centroid in centroids:
                            dist = np.sqrt(np.square(abs(color[0] - centroid[0])) + np.square(abs(color[1] - centroid[1])) + np.square(abs(color[2] - centroid[2])))
                            dists.append(dist)
                        # color appended to min distant cluster
                        clusterNo = dists.index(min(dists))
                        clusters[clusterNo].append(color)
                    print ("original cluster num:", len(clusters))

                    for i in range(len(clusters)):
                        # print("cluster", i, ":", len(clusters[i]))
                        r = 0
                        g = 0
                        b = 0
                        if (len(clusters[i])>0):
                            for col in clusters[i]:
                                r += col[0]
                                g += col[1]
                                b += col[2]
                            r = r/len(clusters[i])
                            g = g/len(clusters[i])
                            b = b/len(clusters[i])
                            newCentroid = [r, g, b]
                            distortion_new += np.sqrt(np.square(abs(centroids[i][0] - newCentroid[0])) + np.square(abs(centroids[i][1] - newCentroid[1])) + np.square(abs(centroids[i][2] - newCentroid[2])))
                            centroids[i] = newCentroid
                        elif(clusters[i] == []):
                            distortion_new += distortion_new/len(clusters)
                        else:
                            pass

                    print("distortion:", distortion)
                    print("distortion_new;", distortion_new)
                    if (iterationNum > 0 and (abs(distortion - distortion_new)) < (distortion * 0.05)):
                        break
                    distortion = distortion_new
                #delete empty clusters
                sortClusters = []
                for i in range(len(clusters)):
                    if (clusters[i] !=  []):
                        sortClusters.append(clusters[i])
                #sort clusters and centroids by length then draw rectangles
                sortClusters = sorted(sortClusters, key=len, reverse=True)
                centroids = list(centroids)
                sortCentroids = []

                clusters_len = []
                sortClusters_len = []

                for num in range(len(clusters)):
                    clusters_len.append(len(clusters[num]))
                    hue_palette_len.append(len(clusters[num]))
                print("##############################################################################")
                for num in range(len(sortClusters)):
                    sortClusters_len.append(len(sortClusters[num]))

                # print("sortClusters", sortClusters)
                print("len(sortClusters):", len(sortClusters))
                print("clusters_len:", clusters_len)
                print("sortClusters_len:", sortClusters_len)

                name_hue = name + str(hue)

                sortCentroids = icon.drawPalette(clusters, sortClusters, clusters_len, sortClusters_len, centroids,
                                                 sortCentroids, name_hue, cluster_palette_folder)
                for hue_centroid in sortCentroids:
                    hue_palette.append(hue_centroid)
                # for top3 in range(3):
                #     hue_palette.append(sortCentroids[top3])
            else:
                pass

        icon.drawPalette_hueUnited(hue_palette, name, palette_folder)

        return hue_palette

    # convert rgb to hsv and then make centroids from each hue
    @classmethod
    def makeClusters_HSVcentroids(self, color_list, name, palette_folder, clusterNum):
        # create 10 hue class
        hue_array = [[] for hue_a in range(10)]
        hue_palette = []
        hue_palette_len = []

        for bgr_color in color_list:
            hsv_color = self.rgb2hsv(bgr_color)
            if (hsv_color[0] < 0):
                hsv_color[0] = 360 - hsv_color[0]
            elif (hsv_color[0] > 360):
                hsv_color[0] = hsv_color[0] % 360

            if (hsv_color[0] == 0 or hsv_color[0] == 360):
                hue_array[0].append(bgr_color)
            else:
                hue_num = int(hsv_color[0] // 36)
                print("hue_num: ", hue_num)
                hue_array[hue_num].append(bgr_color)

        # 初期値の範囲
        centroids = []
        for hue_class in hue_array:
            if hue_class != []:
                centroid_b = 0
                centroid_g = 0
                centroid_r = 0
                for hue_px in hue_class:
                    centroid_b += hue_px[0]
                    centroid_g += hue_px[1]
                    centroid_r += hue_px[2]
                centroid_bgr = [centroid_b / len(hue_class), centroid_g / len(hue_class), centroid_r / len(hue_class)]
                centroids.append(centroid_bgr)
            else:
                pass
        centroids.append([0,0,0])
        centroids.append([255,255,255])

        #number of color for a palette for this building element
        clusterNum = len(centroids)

        print ("centroids:", centroids)
        distortion = 0
        clusters = []
        # make spaces for each clusters
        for i in range(clusterNum):
            clusters.append([])

        # repeat iteration until distortion = distortion*0.005% or after 50 try
        for iterationNum in range(50):
            print("############################################")
            print("element name:", name)
            print("iterationNum:", iterationNum)
            # clusters = []
            distortion_new = 0

            #calc dis from centroids to each color coordinate
            for color in color_list:
                dists = []
                for centroid in centroids:
                    dist = np.sqrt(np.square(abs(color[0] - centroid[0])) + np.square(abs(color[1] - centroid[1])) + np.square(abs(color[2] - centroid[2])))
                    dists.append(dist)
                # color appended to min distant cluster
                clusterNo = dists.index(min(dists))
                # if(clusterNo.type == 'list'):
                #     clusters[clusterNo[0]].append(color)
                # else:
                clusters[clusterNo].append(color)
            print ("original cluster num:", len(clusters))


            for i in range(len(clusters)):
                # print("cluster", i, ":", len(clusters[i]))
                r = 0
                g = 0
                b = 0
                if (len(clusters[i])>0):
                    for col in clusters[i]:
                        r += col[0]
                        g += col[1]
                        b += col[2]
                    r = r/len(clusters[i])
                    g = g/len(clusters[i])
                    b = b/len(clusters[i])
                    newCentroid = [r, g, b]
                    # newCentroid = centroids[i]
                    # distortion_new.append(np.sqrt(np.square(abs(centroids[i][0] - newCentroid[0])) + np.square(abs(centroids[i][1] - newCentroid[1])) + np.square(abs(centroids[i][2] - newCentroid[2]))))
                    distortion_new += np.sqrt(np.square(abs(centroids[i][0] - newCentroid[0])) + np.square(abs(centroids[i][1] - newCentroid[1])) + np.square(abs(centroids[i][2] - newCentroid[2])))
                    centroids[i] = newCentroid
                elif(clusters[i] == []):
                    ##clusters = [x for x in clusters if x]
                    # np.delete(clusters, i, 0)
                    # np.delete(centroids, i, 0)
                    distortion_new += distortion_new/len(clusters)
                    # centroids[i] = centroids[i]
                else:
                    pass

            print("distortion:", distortion)
            print("distortion_new;", distortion_new)
            if (iterationNum > 0 and (abs(distortion - distortion_new)) < (distortion * 0.05)):
                break
            distortion = distortion_new

        # print ("centroids:", centroids)

        #delete empty clusters
        sortClusters = []
        for i in range(len(clusters)):
            if (clusters[i] !=  []):
                sortClusters.append(clusters[i])
        #sort clusters and centroids by length then draw rectangles
        # sortClusters = sortClusters.sort(key=len)
        sortClusters = sorted(sortClusters, key=len, reverse=True)
        # for i in range(len(sortClusters)):
        #     sortClusters[i] = list(sortClusters[i])
        centroids = list(centroids)
        sortCentroids = []

        clusters_len = []
        sortClusters_len = []
        for num in range(len(clusters)):
            # print("len(clusters[" + str(num) + "]:" + str(len(clusters[num])))
            clusters_len.append(len(clusters[num]))
            hue_palette_len.append(len(clusters[num]))
        print("##############################################################################")
        for num in range(len(sortClusters)):
            # print("len(sortClusters[" + str(num) + "]:" + str(len(sortClusters[num])))
            sortClusters_len.append(len(sortClusters[num]))

        # print("sortClusters", sortClusters)
        print("len(sortClusters):", len(sortClusters))
        print("clusters_len:", clusters_len)
        print("sortClusters_len:", sortClusters_len)

        sortCentroids = icon.drawPalette(clusters, sortClusters, clusters_len, sortClusters_len, centroids, sortCentroids, name, palette_folder)

        return sortCentroids, hue_palette_len


    #Color Difference ##############################################################################

    # original thought
    def assign_label(self, pxl, segmented_image, x, y):
        for z, col in enumerate(pxl):
            print(pxl, "pxl")
            if(0 <= col <=63.75):
            #z = 0
                segmented_image[x][y][z] = 0
            elif(63.75 < col <= 127.5):
                #z = 85
                segmented_image[x][y][z] = 85
            elif(127.5 < col <= 191.25):
                #z = 170
                segmented_image[x][y][z] = 170
            elif(191.25 < col <= 255):
                #z = 255
                segmented_image[x][y][z] = 255
            else:
                pass
        return segmented_image

    # Euclidean series
    # Euclidean normal version
    @classmethod
    def euclidean_normal(self, pxl, x, y, z):
        dist = np.sqrt(np.square(abs(pxl[0] - label_centroids[z][0])) +
                       np.square(abs(pxl[1] - label_centroids[z][1])) +
                       np.square(abs(pxl[2] - label_centroids[z][2])))
        return dist

    # Euclidean square value
    @classmethod
    def euclidean_square_value(self, pxl, x, y, z):
        dist = np.sqrt(np.square(abs(pxl[0] - label_normal[z][0])) + np.square(abs(pxl[1] - label_normal[z][1])) + np.square(
                abs(pxl[2] - label_normal[z][2])))
        dist = np.sqrt(np.square(abs(np.square(pxl[0]) - np.square(label_centroids[z][0]))) +
                       np.square(abs(np.square(pxl[1]) - np.square(label_centroids[z][1]))) +
                       np.square(abs(np.square(pxl[2]) - np.square(label_centroids[z][2]))))
        return  dist

    # Euclidean with weights red 30%, green 59%, and blue 11%
    @classmethod
    def euclidean_305911(self, pxl, x, y, z):
        dist = np.sqrt(np.square(abs(pxl[0] - label_centroids[z][0])) * 0.11 +
                       np.square(abs(pxl[1] - label_centroids[z][1])) * 0.59 +
                       np.square(abs(pxl[2] - label_centroids[z][2])) * 0.30)
        return dist

    # Euclidean with closer approximations would be more properly coefficients of 2, 4, and 3
    @classmethod
    def enclidean_243(self, pxl, x, y, z):
        dist = np.sqrt(np.square(abs(pxl[0] - label_centroids[z][0])) * 3 +
                       np.square(abs(pxl[1] - label_centroids[z][1])) * 4 +
                       np.square(abs(pxl[2] - label_centroids[z][2])) * 2)
        return dist

    @classmethod #used
    # Euclidean with one of the better low-cost approximations (using a color range of 0–255)******
    def euclidean_lowcost(self, pxl, x, y, z):
        r = (pxl[2] + label_centroids[z][2]) * 1.0 / 2
        rr = r * (abs(pxl[2] - label_centroids[z][2]) - abs(pxl[0] - label_centroids[z][0])) * 1.0 / 256
        dist = np.sqrt(np.square(abs(pxl[0] - label_centroids[z][0])) * 3 +
                       np.square(abs(pxl[1] - label_centroids[z][1])) * 4 +
                       np.square(abs(pxl[2] - label_centroids[z][2])) * 2 +
                       rr)
        return dist

    # CIE series
    def _conv_func(self, v):
        if v > 0.008856:
            return v ** (1.0 / 3.0)
        else:
            return (903.3 * v + 16) / 116.0

    @classmethod
    def rgb2lab(self, inputColor):

        num = 0
        RGB = [0, 0, 0]

        for value in inputColor:
            value = float(value) / 255

            if value > 0.04045:
                value = ((value + 0.055) / 1.055) ** 2.4
            else:
                value = value / 12.92

            RGB[num] = value * 100
            num = num + 1

        XYZ = [0, 0, 0, ]

        X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
        Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
        Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
        XYZ[0] = round(X, 4)
        XYZ[1] = round(Y, 4)
        XYZ[2] = round(Z, 4)

        XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
        XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
        XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883

        num = 0
        for value in XYZ:

            if value > 0.008856:
                value = value ** (0.3333333333333333)
            else:
                value = (7.787 * value) + (16 / 116)

            XYZ[num] = value
            num = num + 1

        Lab = [0, 0, 0]

        L = (116 * XYZ[1]) - 16
        a = 500 * (XYZ[0] - XYZ[1])
        b = 200 * (XYZ[1] - XYZ[2])

        Lab[0] = round(L, 4)
        Lab[1] = round(a, 4)
        Lab[2] = round(b, 4)

        return Lab

    @classmethod
    def rgb2hsi(self, bgr): #from automatic color palette
        bgr = [float(bgr[0]) * 1.0 / 255, float(bgr[1]) * 1.0 / 255, float(bgr[2]) * 1.0 / 255]
        hsi_I = (bgr[2] + bgr[1] + bgr[0])/3
        hsi_S = np.sqrt(np.square(bgr[2]-hsi_I) + np.square(bgr[1]-hsi_I) + np.square(bgr[0]-hsi_I))
        hsi_H = np.arccos(((bgr[1]-hsi_I)-(bgr[0]-hsi_I))/(hsi_S*np.sqrt(2)))
        hsi = [hsi_H, hsi_S, hsi_I]
        return  hsi

    @classmethod
    def rgb2hsv(self, bgr): #wikipedia
        # R, G, Bの値を取得して0～1の範囲内にする
        bgr_1 = [float(bgr[0])*1.0/255, float(bgr[1])*1.0/255, float(bgr[2])*1.0/255]
        print("BGR/255: ", bgr_1)

        # R, G, Bの値から最大値と最小値を計算
        mx, mn = max(bgr_1[2], bgr_1[1], bgr_1[0]), min(bgr_1[2], bgr_1[1], bgr_1[0])

        # 最大値 - 最小値
        diff = mx - mn

        # Hの値を計算
        if mx == mn:
            hsv_h = 0
        elif mx == bgr_1[2]:
            hsv_h = 60 * ((bgr_1[1] - bgr_1[0])*1.0 / diff)
        elif mx == bgr_1[1]:
            hsv_h = 60 * ((bgr_1[0] - bgr_1[2])*1.0 / diff) + 120
        elif mx == bgr_1[0]:
            hsv_h = 60 * ((bgr_1[2] - bgr_1[1])*1.0 / diff) + 240
        if hsv_h < 0: hsv_h = hsv_h + 360

        # Sの値を計算
        if mx != 0:
            hsv_s = diff / mx
        else:
            hsv_s = 0

        # Vの値を計算
        hsv_v = mx

        # Hを0～179, SとVを0～255の範囲の値に変換
        # hsv = [hsv_h * 0.5, hsv_s * 255, hsv_v * 255]

        hsv = [hsv_h, hsv_s, hsv_v]

        return hsv

    # CIE 94 -----> wrong calcuration
    def cie94(self, pxl, x, y, z):
        pxl_rgb = [pxl[2], pxl[1], pxl[0]]
        label_rgb = [label_centroids[z][2], label_centroids[z][1], label_centroids[z][0]]
        pxl_lab = Cluster.rgb2lab(pxl_rgb)
        label_lab = Cluster.rgb2lab(label_rgb)
        # print ("pxl_lab", pxl_lab)
        # print(("label_lab", label_lab))

        # weightingK = [1, 0,045, 0,015] #graphic arts
        weightingK = [2, 0.048, 0.014] #textiles
        l_diff = abs(pxl_lab[0] - label_lab[0])
        c_diff = abs(np.sqrt(np.square(abs(pxl_lab[1] + pxl_lab[2]))) - np.sqrt(np.square(abs(label_lab[1] + label_lab[2]))))
        h_diff = np.sqrt(np.square(abs(pxl_lab[1] + label_lab[1])) + np.square(abs(pxl_lab[2] + label_lab[2])) - np.square(c_diff))
        sl = 1
        sc = 1 + weightingK[1]*np.sqrt(np.square(abs(pxl_lab[1] + pxl_lab[2])))
        sh = 1 + weightingK[1]*np.sqrt(np.square(abs(label_lab[1] + label_lab[2])))

        dist = np.sqrt(np.square(abs(l_diff*1.0/(weightingK[0]*sl))) +
                       np.square(abs(c_diff*1.0/weightingK[1]*sc)) +
                       np.square(abs(h_diff*1.0/weightingK[2]*sh)))

        return dist

    @classmethod
    def cie2000(self, lab0, lab1):
        # 1.Calculate C0 C1 h0 h1
        C0ab = math.sqrt(math.pow(lab0[1], 2) + math.pow(lab0[2], 2))
        C1ab = math.sqrt(math.pow(lab1[1], 2) + math.pow(lab1[2], 2))
        C01ab = (C0ab + C1ab) * 1.0 / 2
        G = 0.5 * (1 - math.sqrt(math.pow(C01ab, 7) / math.pow(C01ab, 7) + math.pow(25, 7)))
        a0dash = (1 + G) * lab0[1]
        a1dash = (1 + G) * lab1[1]
        C0dash = math.sqrt(math.pow(a0dash, 2) + math.pow(lab0[2], 2))
        C1dash = math.sqrt(math.pow(a1dash, 2) + math.pow(lab1[2], 2))
        if lab0[1] == lab0[2]:
            h0 = 0
        else:
            h0 = math.degrees(math.atan2(lab0[2], a0dash))
        if lab1[1] == lab1[2]:
            h1 = 0
        else:
            h1 = math.degrees(math.atan2(lab1[2], a1dash))

        # 2. Calculate L, C, H
        Ldiff = lab1[0] - lab1[0]
        Cdiff = C1dash - C0dash
        if C0dash == 0 or C1dash == 0:
            hdiff = 0
        elif C0dash * C1dash != 0 and abs(h1 - h0) <= 180:
            hdiff = h1 - h0
        elif C0dash * C1dash != 0 and (h1 - h0) > 180:
            hdiff = (h1 - h0) - 360
        elif C0dash * C1dash != 0 and (h1 - h0) < -180:
            hdiff = (h1 - h0) + 360
        else:
            pass
        Hdifference = 2 * math.sqrt(C0ab * C1ab) * math.sin(hdiff * 1.0 / 2)

        # 3. Calculate CIEDE2000 colordiffE
        Lmean = (lab1[0] + lab1[0]) * 1.0 / 2
        Cmean = (C1dash - C0dash) * 1.0 / 2
        if C0dash * C1dash != 0 and abs(h0 - h1) <= 180:
            hmeans = (h0 + h1) * 1.0 / 2
        elif C0dash * C1dash != 0 and abs(h0 - h1) > 180 and (h0 + h1) < 360:
            hmeans = (h0 + h1 + 360) * 1.0 / 2
        elif C0dash * C1dash != 0 and abs(h0 - h1) > 180 and (h0 + h1) >= 360:
            hmeans = (h0 + h1 - 360) * 1.0 / 2
        elif C0dash == 0 or C1dash == 0:
            hmeans = (h0 + h1)
        else:
            pass
        T_coldiff = 1 - 0.17 * math.cos(hmeans - 30) + 0.24 * math.cos(2 * hmeans) + 0.32 * math.cos(
            3 * hmeans + 6) - 0.2 * math.cos(4 * hmeans - 63)
        thetadiff = 30 * math.exp(-1.0 * math.pow((hmeans - 275), 2) / 25)
        Rc_coldiff = 2 * (math.sqrt(math.pow(Cmean, 7) / math.pow(Cmean, 7) + math.pow(25, 7)))
        sL_coldiff = 1 + (0.015 * math.pow((Lmean - 50), 2)) * 1.0 / math.sqrt(20 + math.pow((Lmean - 50), 2))
        sC_coldiff = 1 + 0.045 * Cmean
        sH_coldiff = 1 + 0.015 * Cmean * T_coldiff
        Rt_coldiff = -1.0 * math.sin(2 * thetadiff) * Rc_coldiff
        # kL = kC = kH, parametric weighting factors
        kL_coldiff = 1
        kC_coldiff = kL_coldiff
        kH_coldiff = kL_coldiff
        E_0 = math.pow((Ldiff * 1.0 / (kL_coldiff * sL_coldiff)), 2)
        E_1 = math.pow((Cdiff * 1.0 / (kC_coldiff * sC_coldiff)), 2)
        E_2 = math.pow((Hdifference * 1.0 / (kH_coldiff * sH_coldiff)), 2)
        E_3 = Rt_coldiff * (Cdiff * 1.0 / (kC_coldiff * sC_coldiff)) * ((Hdifference * 1.0) / (kH_coldiff * sH_coldiff))
        print("E_0, E_1, E_2, E_3: ", E_0, "/", E_1, "/", E_2, "/", E_3)
        colordiffE = math.sqrt( abs(E_0 + E_1 + E_2 + E_3) )
        return colordiffE

    ##############################################################################

if __name__ == '__main__':

    print("##############################################################################")

    # cluster = Cluster()

    for c in range(len(label_centroids)-1):
        # hsi = Cluster.rgb2hsi(color)
        # hsv = Cluster.rgb2hsv(color)
        # print("HSI: ", hsi, " / HSV: ", hsv)
        lab0 = Cluster.rgb2lab(label_centroids[c])
        lab1 = Cluster.rgb2lab(label_centroids[c+1])
        distance = Cluster.cie2000(lab0, lab1)
        print("Lab0*: ", lab0, "/ Lab1*: ", lab1)
        print("distance: ", distance)