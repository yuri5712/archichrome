import os
import math
import numpy as np
import cv2
import itertools
import random
import clustering

errorValue = [[-1]]  # was [[-1, -1, -1]]
#
elementsNames = ["no_label", "background", "window", "door", "sill", "balcony", "shop"]
# elementsCentroid = [[[74, 31, 35], [110, 59, 61], [124, 91, 90], [195, 219, 242], [173, 193, 224], [149, 133, 135]],
#                     [[113, 66, 67], [121, 114, 111], [146, 81, 85], [63, 47, 48], [153, 146, 144], [172, 190, 218],
#                      [177, 165, 163], [195, 211, 237]],
#                     [[119, 134, 135], [103, 106, 113], [130, 60, 64], [54, 46, 56], [183, 189, 211]],
#                     [[151, 186, 223], [113, 122, 145], [118, 144, 174], [60, 64, 79], [137, 169, 203],
#                      [90, 97, 119], [109, 68, 74]],
#                     [[129, 67, 72], [182, 201, 228], [170, 175, 184], [130, 130, 142], [54, 36, 45],
#                      [165, 147, 148], [76, 66, 90]],
#                     [[135, 68, 72], [174, 175, 171]],
#                     [[138, 64, 68], [116, 112, 105], [162, 156, 150], [82, 40, 42], [195, 210, 230]]
#                     ]
elementsNamesAll = ["no_label", "background", "facade", "window", "door", "cornice", "sill", "balcony", "blind", "deco", "molding", "pillar", "shop"]
elementsColor = [[[74, 31, 35], [110, 59, 61], [124, 91, 90], [195, 219, 242], [173, 193, 224], [149, 133, 135]],
                [[113, 66, 67], [121, 114, 111], [146, 81, 85], [63, 47, 48], [153, 146, 144], [172, 190, 218],
                     [177, 165, 163], [195, 211, 237]],
                [[-1, -1, -1]],
                [[119, 134, 135], [103, 106, 113], [130, 60, 64], [54, 46, 56], [183, 189, 211]],
                [[151, 186, 223], [113, 122, 145], [118, 144, 174], [60, 64, 79], [137, 169, 203],
                     [90, 97, 119], [109, 68, 74]],
                [[-1, -1, -1]],
                [[129, 67, 72], [182, 201, 228], [170, 175, 184], [130, 130, 142], [54, 36, 45],
                     [165, 147, 148], [76, 66, 90]],
                [[135, 68, 72], [174, 175, 171]],
                [[-1, -1, -1]],
                [[-1, -1, -1]],
                [[-1, -1, -1]],
                [[-1, -1, -1]],
                [[138, 64, 68], [116, 112, 105], [162, 156, 150], [82, 40, 42], [195, 210, 230]]
                ]

# variables
size = 10
iconSize = size*19
shadeSize = 5
frame = 70
margin = 20
numColor = 2
numElement = len(elementsNames)
shadeColor = [235, 235, 235]
shadeColorW = [255, 255, 255]
patternNum = 25
alpha = 0.2

# Color Palettes
def drawPalette(clusters, sortClusters, clusters_len,sortClusters_len, centroids, sortCentroids, name, palette_folder):
    # draw a palette
    paletteH = 50
    paletteW = 50

    # background of the palette
    palette = np.full((paletteH, paletteW * len(sortClusters), 3), 250, dtype=np.uint8)
    # palette2 = np.full((paletteH, paletteW * len(clusters), 3), 250, dtype=np.uint8)

    # pick one color if the number of pixcel is same for several colors
    count = 0
    indexList = []
    usedIndex = []

    for i in range(len(sortClusters)):
        if clusters_len.count(sortClusters_len[i]) == 1:
            sortCentroids.append(centroids[clusters.index(sortClusters[i])])
        elif clusters_len.count(sortClusters_len[i]) > 1 and clusters_len.count(sortClusters_len[i]) == count - 1:
            count = 0
            usedIndex = []
            indexList = [j for j, x in enumerate(clusters_len) if x == sortClusters_len[i]]
            if indexList[count] not in usedIndex:
                sortCentroids.append(centroids[indexList[count]])
                usedIndex.append(indexList[count])
                count += 1
            else:
                pass
        elif clusters_len.count(sortClusters_len[i]) > 1:
            indexList = [j for j, x in enumerate(clusters_len) if x == sortClusters_len[i]]
            if indexList[count] not in usedIndex:
                sortCentroids.append(centroids[indexList[count]])
                usedIndex.append(indexList[count])
                count += 1
            else:
                pass

        else:
            pass

        # print("indexList:", indexList)
        # print("usedIndex:", usedIndex)
        # print("count:", count)
        sortCentroids[i] = list(sortCentroids[i])
        cv2.rectangle(palette, (paletteW * i, 0), (paletteW * (i + 1), paletteH),
                      (int(sortCentroids[i][2]), int(sortCentroids[i][1]), int(sortCentroids[i][0])), -1)  # GBR

    cv2.imwrite(os.path.join(palette_folder, name + '.jpg'), palette)
    print("sortCentroid : ", sortCentroids)

    print("icon created")
    return sortCentroids

def drawPalette_hueUnited(hue_palette, name, palette_folder):
    # draw a palette
    paletteH = 50
    paletteW = 50

    # background of the palette
    palette = np.full((paletteH, paletteW * len(hue_palette), 3), 250, dtype=np.uint8)
    for i in range(len(hue_palette)):
        hue_palette[i] = list(hue_palette[i])
        cv2.rectangle(palette, (paletteW * i, 0), (paletteW * (i + 1), paletteH),
                      (int(hue_palette[i][2]), int(hue_palette[i][1]), int(hue_palette[i][0])), -1)  # GBR
        # cv2.rectangle(palette, (paletteW * i, 0), (paletteW * (i + 1), paletteH),
        #               (int(hue_palette[i][0]), int(hue_palette[i][1]), int(hue_palette[i][2])), -1)  # GBR

    cv2.imwrite(os.path.join(palette_folder, name + '_hue.jpg'), palette)
    print("hue united palette created")


#call selection of patterns
# using this random
def pattern_random(elementsColor, elementsNames, palette_folder, elements_len):

    colorPatterns = []
    patterns = []
    for i in range(patternNum):
        selection =[]
        for element in elementsColor:
            selection.append(random.choice(element))
        patterns.append(selection)

    draw = drawIcon_original(patterns, colorPatterns, palette_folder)
    # drawIcon_emptyShade(patterns, colorPatterns, palette_folder)

def pattern_random_hatch(elementsColor, elementsNames, palette_folder, elements_len):

    colorPatterns = []
    patterns = []
    for i in range(patternNum):
        selection =[]
        for element in elementsColor:
            selection.append(random.choice(element))
        patterns.append(selection)

    # drawIcon_original(patterns, colorPatterns, palette_folder)
    print("random patterns created")

    # shade = drawIcon_emptyShade(patterns, palette_folder)
    # draw = drawIcon_emptyShade_for(patterns, palette_folder)
    white = drawIcon_emptyShade_white(patterns, palette_folder)


def drawIcon(elementsColor, elementsNames, palette_folder, elements_len):

    colorPatterns = []

    #prepare for itertool to make the list length == numColor & elements which is not drawn in icon --> errorValue
    for i,element in enumerate(elementsColor):
        elementColor = []
        if(elementsNamesAll[i] == "no_label" or elementsColor[0] == [-1, -1, -1]):
            elementColor.append([-1, -1, -1])
            colorPatterns.append(elementColor)
        else:
            # to make elementsColor num = numColor(2), @max2
            if (len(element) >= numColor):
                for j in range(numColor):
                    elementColor.append(element[j])
                colorPatterns.append(elementColor)
            elif len(element) < numColor:
                for j in range(len(element)):
                    elementColor.append(element[j])
                colorPatterns.append(elementColor)
            else:
                pass


    # color pattern generator
    patterns = list(itertools.product(colorPatterns[0], colorPatterns[1], colorPatterns[2], colorPatterns[3], colorPatterns[4], colorPatterns[5], colorPatterns[6],
                                      colorPatterns[7], colorPatterns[8], colorPatterns[9], colorPatterns[10], colorPatterns[11], colorPatterns[12]))
    draw = drawIcon_original(patterns, colorPatterns, palette_folder)

def drawIcon_line(elementsColor, elementsNames, palette_folder, elements_len):

    colorPatterns = []

    #prepare for itertool to make the list length == numColor & elements which is not drawn in icon --> errorValue
    for i,element in enumerate(elementsColor):
        elementColor = []
        if(elementsNamesAll[i] == "no_label" or elementsColor[0] == [-1, -1, -1]):
            elementColor.append([-1, -1, -1]) # if numColor=2
            elementColor.append([-1, -1, -1])
            colorPatterns.append(elementColor)
        else:
            # to make elementsColor num = numColor(2), @max2
            if (len(element) >= numColor):
                for j in range(numColor):
                    elementColor.append(element[j])
                colorPatterns.append(elementColor)
            elif len(element) < numColor:
                for j in range(len(element)):
                    elementColor.append(element[j])
                colorPatterns.append(elementColor)
            else:
                pass


    # color pattern generator
    patterns = list(itertools.product(colorPatterns[0], colorPatterns[1], colorPatterns[2], colorPatterns[3], colorPatterns[4], colorPatterns[5], colorPatterns[6],
                                      colorPatterns[7], colorPatterns[8], colorPatterns[9], colorPatterns[10], colorPatterns[11], colorPatterns[12]))

    draw = drawIcon_original(patterns, colorPatterns, palette_folder)

def drawIcon_minmaxNum(elementsColor, elementsNames, palette_folder, elements_len):

    colorPatterns = []

    #get max min
    for i,element in enumerate(elementsColor):
        if(elementsNamesAll[i] == "no_label" or elementsColor[0] == [-1, -1, -1]):
            elementColor = [-1, -1, -1]
            colorPatterns.append(elementColor)
        else:
            elementColor = select_maxminNum(element, elements_len[i])
            colorPatterns.append(elementColor)

    # color pattern generator
    patterns = list(itertools.product(colorPatterns[0], colorPatterns[1], colorPatterns[2], colorPatterns[3], colorPatterns[4], colorPatterns[5], colorPatterns[6],
                                      colorPatterns[7], colorPatterns[8], colorPatterns[9], colorPatterns[10], colorPatterns[11], colorPatterns[12]))
    draw = drawIcon_original(patterns, colorPatterns, palette_folder)

def drawIcon_minmaxDist(elementsColor, elementsNames, palette_folder, elements_len):

    colorPatterns = []

    #calc distance
    for i,element in enumerate(elementsColor):
        if(elementsNamesAll[i] == "no_label" or elementsColor[0] == [-1, -1, -1]):
            elementColor = [-1, -1, -1]
            colorPatterns.append(elementColor)
        else:
            elementColor = []
            colorPatterns.append(elementColor)

    # color pattern generator
    patterns = list(itertools.product(colorPatterns[0], colorPatterns[1], colorPatterns[2], colorPatterns[3], colorPatterns[4], colorPatterns[5], colorPatterns[6],
                                      colorPatterns[7], colorPatterns[8], colorPatterns[9], colorPatterns[10], colorPatterns[11], colorPatterns[12]))

    draw = drawIcon_original(patterns, colorPatterns, palette_folder)


# draw rectangles and empty sign
def drawIcon_original(patterns, colorPatterns, palette_folder):

    # draw canvas
    # iconMatrixNum = math.ceil(math.sqrt(len(patterns)))
    iconMatrixNum = math.floor(math.sqrt(len(patterns)))
    canvasSize = iconMatrixNum * iconSize + (iconMatrixNum + 1) * margin
    mod = math.pow(iconMatrixNum, 2)-len(patterns)
    if(mod < 0):
        canvasSizeH = (iconMatrixNum + 1)* iconSize + (iconMatrixNum + 2) * margin
    else:
        canvasSizeH = (iconMatrixNum)* iconSize + (iconMatrixNum + 1) * margin

    canvas = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)

    # print("colorPatterns: ", colorPatterns)
    # print("len(pattern): ", len(patterns))
    #
    # print("pattern[0]: ", patterns[0])
    # print("len(pattern[0]): ", len(patterns[0]))
    # print("math.ceil(19/2): ", math.ceil(19/2))
    # print("19//2: ", 19//2)
    #
    # print("iconMatrixNum: ", iconMatrixNum)
    # print("canvasSize: ", canvasSize)
    # print("canvasSizeH: ", canvasSizeH)
    # print("mod:", mod)

    # draw each icons
    for p,pattern in enumerate(patterns):

        # print("pattern[" + str(p)+ "]: " + str(pattern[1][0]) + str(pattern[1][1]) + str(pattern[1][2]))

        #get the position of patterns[p]
        if(p == 0):
            iconRowNum = (p+1)//iconMatrixNum
        else:
            iconRowNum = p//iconMatrixNum
        iconColumnNum = p % iconMatrixNum

        # calc start point(startX, startY) of icon No.[i]
        startX = (iconColumnNum+1)*margin + iconColumnNum*iconSize
        startY = (iconRowNum+1)*margin + iconRowNum*iconSize

        # draw transparent pattern
        shadeNum =int(iconSize/shadeSize)
        for r in range(shadeNum):
            if (r%2 == 0):
                for c in range(math.ceil((shadeNum)/2)):
                    cv2.rectangle(canvas, (startX + 2*c*shadeSize, startY + r*shadeSize), (startX + (2*c*shadeSize + shadeSize), startY + (r*shadeSize+shadeSize)), (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
            elif (r%2 == 1):
                for c in range((shadeNum)//2):
                    cv2.rectangle(canvas, (startX + (2*c*shadeSize + shadeSize), startY + r*shadeSize), (startX + (2*c*shadeSize + 2*shadeSize), startY + (r*shadeSize+shadeSize)), (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
            else:
                pass

        # draw each shapes

        # 1: background
        if(pattern[1][0] == -1):
            pass
        elif(pattern[1][0] != -1):
            cv2.rectangle(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size), (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size), (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size), (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)

        # 2: facade
        if(pattern[2][0] == -1):
            pass
        elif (pattern[2][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size), (int(pattern[2][2]), int(pattern[2][1]), int(pattern[2][0])), thickness=-1)

        # 3: window
        if(pattern[3][0] == -1):
            pass
        elif (pattern[3][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size), (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size), (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size), (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)

        # 4: door
        if(pattern[4][0] == -1):
            pass
        elif (pattern[4][0] != -1):
            cv2.rectangle(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size), (int(pattern[4][2]), int(pattern[4][1]), int(pattern[4][0])), thickness=-1)

        # 8: blind
        if(pattern[8][0] == -1):
            pass
        elif (pattern[8][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 4 * size, startY + 11 * size), (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 18 * size), (startX + 4 * size, startY + 18 * size), (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size), (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 18 * size), (startX + 8 * size, startY + 18 * size), (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 12 * size, startY + 11 * size), (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 15 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size), (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)

        # 5: cornice
        if(pattern[5][0] == -1):
            pass
        elif(pattern[5][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size), (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 15 * size), (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size), (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)

        # 6: sill
        if(pattern[6][0] == -1):
            pass
        elif(pattern[6][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size), (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 17 * size), (startX + 8 * size, startY + 18 * size), (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size), (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)

        # 7: balcony
        if(pattern[7][0] == -1):
            pass
        elif(pattern[7][0] != -1):
            cv2.rectangle(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size), (int(pattern[7][2]), int(pattern[7][1]), int(pattern[7][0])), thickness=-1)

        # 9: deco
        if(pattern[9][0] == -1):
            pass
        elif(pattern[9][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 18 * size, startY + 6 * size), (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 1 * size, startY + 12 * size), (startX + 18 * size, startY + 13 * size), (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)

        # 10: molding
        if(pattern[10][0] == -1):
            pass
        elif(pattern[10][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size), (int(pattern[10][2]), int(pattern[10][1]), int(pattern[10][0])), thickness=-1)

        # 11: pillar
        if(pattern[11][0] == -1):
            pass
        elif(pattern[11][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size), (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size), (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size), (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)

        # 12: shop
        if(pattern[12][0] == -1):
            pass
        elif(pattern[12][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size), (int(pattern[12][2]), int(pattern[12][1]), int(pattern[12][0])), thickness=-1)

    print("##############################################################################")


    cv2.imwrite(os.path.join(palette_folder, 'icons_random_'+ str(numColor) + '^' + str(numElement) + '.jpg'), canvas)
    print("Archichrome created")

def drawIcon_emptyLine(patterns, palette_folder):
    # draw canvas
    # iconMatrixNum = math.ceil(math.sqrt(len(patterns)))

    iconMatrixNum = math.floor(math.sqrt(len(patterns)))
    canvasSize = iconMatrixNum * iconSize + (iconMatrixNum + 1) * margin
    mod = math.pow(iconMatrixNum, 2)-len(patterns)
    if(mod < 0):
        canvasSizeH = (iconMatrixNum + 1)* iconSize + (iconMatrixNum + 2) * margin
    else:
        canvasSizeH = (iconMatrixNum)* iconSize + (iconMatrixNum + 1) * margin

    canvas = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)

    print("elementsColor: ", elementsColor)

    # draw each icons
    for p, pattern in enumerate(patterns):

        # get the position of patterns[p]
        if (p == 0):
            iconRowNum = (p + 1) // iconMatrixNum
        else:
            iconRowNum = p // iconMatrixNum
        iconColumnNum = p % iconMatrixNum

        # calc start point(startX, startY) of icon No.[i]
        startX = (iconColumnNum + 1) * margin + iconColumnNum * iconSize
        startY = (iconRowNum + 1) * margin + iconRowNum * iconSize


        # 1: background
        if (pattern[1][0] == -1):
            cv2.rectangle(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])))
            cv2.line(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size), (0, 0, 0), thickness=1,
                     lineType=cv2.LINE_8)
            cv2.line(canvas, (startX, startY + 1 * size), (startX + 19 * size, startY), (0, 0, 0), thickness=1,
                     lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])))
            cv2.line(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX, startY + 19 * size), (startX + 1 * size, startY + 1 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])))
            cv2.line(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 18 * size, startY + 19 * size), (startX + 19 * size, startY + 1 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[1][0] != -1):
            cv2.rectangle(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)

        # 2: facade
        if (pattern[2][0] == -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[2][2]), int(pattern[2][1]), int(pattern[2][0])))
            cv2.line(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 1 * size, startY + 19 * size), (startX + 18 * size, startY + 1 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[2][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[2][2]), int(pattern[2][1]), int(pattern[2][0])), thickness=-1)

        # 3: window
        if (pattern[3][0] == -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 11 * size), (startX + 8 * size, startY + 7 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])))
            cv2.line(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 11 * size, startY + 11 * size), (startX + 16 * size, startY + 7 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 18 * size), (startX + 8 * size, startY + 14 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[3][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)

        # 4: door
        if (pattern[4][0] == -1):
            cv2.rectangle(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size),
                          (int(pattern[4][2]), int(pattern[4][1]), int(pattern[4][0])))
            cv2.line(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 12 * size, startY + 19 * size), (startX + 15 * size, startY + 14 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[4][0] != -1):
            cv2.rectangle(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size),
                          (int(pattern[4][2]), int(pattern[4][1]), int(pattern[4][0])), thickness=-1)

        # 8: blind
        if (pattern[8][0] == -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 4 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 7 * size), (startX + 4 * size, startY + 11 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 11 * size), (startX + 4 * size, startY + 7 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 4 * size, startY + 18 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 14 * size), (startX + 4 * size, startY + 18 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 18 * size), (startX + 4 * size, startY + 14 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 7 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])))
            cv2.line(canvas, (startX + 7 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 7 * size, startY + 11 * size), (startX + 8 * size, startY + 7 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 7 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])))
            cv2.line(canvas, (startX + 7 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 7 * size, startY + 18 * size), (startX + 8 * size, startY + 14 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 12 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])))
            cv2.line(canvas, (startX + 11 * size, startY + 7 * size), (startX + 12 * size, startY + 11 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 11 * size, startY + 11 * size), (startX + 12 * size, startY + 7 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 15 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])))
            cv2.line(canvas, (startX + 15 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 15 * size, startY + 11 * size), (startX + 16 * size, startY + 7 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[8][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 4 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 4 * size, startY + 18 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 12 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 15 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)

        # 5: cornice
        if (pattern[5][0] == -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 8 * size), (startX + 8 * size, startY + 7 * size), (0, 0, 0),
                     thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 15 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 15 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 15 * size), (startX + 8 * size, startY + 14 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])))
            cv2.line(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 11 * size, startY + 8 * size), (startX + 16 * size, startY + 7 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[5][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 15 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)

        # 6: sill
        if (pattern[6][0] == -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 11 * size), (startX + 8 * size, startY + 10 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 3 * size, startY + 17 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])))
            cv2.line(canvas, (startX + 3 * size, startY + 17 * size), (startX + 8 * size, startY + 18 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 3 * size, startY + 18 * size), (startX + 8 * size, startY + 17 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])))
            cv2.line(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 11 * size, startY + 11 * size), (startX + 16 * size, startY + 10 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[6][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 17 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)

        # 7: balcony
        if (pattern[7][0] == -1):
            cv2.rectangle(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size),
                          (int(pattern[7][2]), int(pattern[7][1]), int(pattern[7][0])))
            cv2.line(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 10 * size, startY + 12 * size), (startX + 17 * size, startY + 10 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[7][0] != -1):
            cv2.rectangle(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size),
                          (int(pattern[7][2]), int(pattern[7][1]), int(pattern[7][0])), thickness=-1)

        # 9: deco
        if (pattern[9][0] == -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 18 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])))
            cv2.line(canvas, (startX + 1 * size, startY + 5 * size), (startX + 18 * size, startY + 6 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 1 * size, startY + 6 * size), (startX + 18 * size, startY + 5 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 1 * size, startY + 12 * size), (startX + 18 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])))
            cv2.line(canvas, (startX + 1 * size, startY + 12 * size), (startX + 18 * size, startY + 13 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 1 * size, startY + 13 * size), (startX + 18 * size, startY + 12 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[9][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 18 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 1 * size, startY + 12 * size), (startX + 18 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)

        # 10: molding
        if (pattern[10][0] == -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size),
                          (int(pattern[10][2]), int(pattern[10][1]), int(pattern[10][0])), thickness=-1)
            cv2.line(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 1 * size, startY + 5 * size), (startX + 18 * size, startY + 1 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[10][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size),
                          (int(pattern[10][2]), int(pattern[10][1]), int(pattern[10][0])), thickness=-1)

        # 11: pillar
        if (pattern[11][0] == -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])))
            cv2.line(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 1 * size, startY + 19 * size), (startX + 2 * size, startY + 5 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])))
            cv2.line(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 9 * size, startY + 19 * size), (startX + 10 * size, startY + 5 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

            cv2.rectangle(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])))
            cv2.line(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 17 * size, startY + 19 * size), (startX + 18 * size, startY + 5 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)

        elif (pattern[11][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)

        # 12: shop
        if (pattern[12][0] == -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size),
                          (int(pattern[12][2]), int(pattern[12][1]), int(pattern[12][0])))
            cv2.line(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
            cv2.line(canvas, (startX + 2 * size, startY + 19 * size), (startX + 9 * size, startY + 13 * size),
                     (0, 0, 0), thickness=1, lineType=cv2.LINE_8)


        elif (pattern[12][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size),
                          (int(pattern[12][2]), int(pattern[12][1]), int(pattern[12][0])), thickness=-1)

    print("##############################################################################")

    cv2.imwrite(os.path.join(palette_folder, 'icons_max2_' + str(numColor) + '^' + str(numElement) + '_lined.jpg'), canvas)

    print("Archichrome created")

# using this
def drawIcon_emptyShade(patterns, palette_folder):
    # draw canvas
    # iconMatrixNum = math.ceil(math.sqrt(len(patterns)))

    iconMatrixNum = math.floor(math.sqrt(len(patterns)))
    canvasSize = iconMatrixNum * iconSize + (iconMatrixNum + 1) * margin
    mod = math.pow(iconMatrixNum, 2)-len(patterns)
    if(mod < 0):
        canvasSizeH = (iconMatrixNum + 1)* iconSize + (iconMatrixNum + 2) * margin
    else:
        canvasSizeH = (iconMatrixNum)* iconSize + (iconMatrixNum + 1) * margin

    canvas = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)
    transparent_hatch = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)

    # print("elementsColor: ", elementsColor)

    # draw each icons
    for p, pattern in enumerate(patterns):
        # get the position of patterns[p]
        if (p == 0):
            iconRowNum = (p + 1) // iconMatrixNum
        else:
            iconRowNum = p // iconMatrixNum
        iconColumnNum = p % iconMatrixNum

        # calc start point(startX, startY) of icon No.[i]
        startX = (iconColumnNum + 1) * margin + iconColumnNum * iconSize
        startY = (iconRowNum + 1) * margin + iconRowNum * iconSize


        # 1: background
        if (pattern[1][0] == -1):
            transparent_hatch(startX, startY, startX + 19 * size, startY + 1 * size, transparent_hatch)
            transparent_hatch(startX, startY + 1 * size, startX + 1 * size, startY + 19 * size, transparent_hatch)
            transparent_hatch(startX + 18 * size, startY + 1 * size, startX + 1 * size, startY + 1 * size, transparent_hatch)

        elif (pattern[1][0] != -1):
            cv2.rectangle(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)

        # 2: facade
        if (pattern[2][0] == -1):
            transparent_hatch(startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 19 * size, transparent_hatch)

        elif (pattern[2][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[2][2]), int(pattern[2][1]), int(pattern[2][0])), thickness=-1)

        # 3: window
        if (pattern[3][0] == -1):
            transparent_hatch(startX + 3 * size, startY + 7 * size, startX + 8 * size, startY + 11 * size, transparent_hatch)
            transparent_hatch(startX + 11 * size, startY + 7 * size, startX + 16 * size, startY + 11 * size, transparent_hatch)
            transparent_hatch(startX + 3 * size, startY + 14 * size, startX + 8 * size, startY + 18 * size, transparent_hatch)

        elif (pattern[3][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)

        # 4: door
        if (pattern[4][0] == -1):
            transparent_hatch(startX + 12 * size, startY + 14 * size, startX + 15 * size, startY + 19 * size, transparent_hatch)

        elif (pattern[4][0] != -1):
            cv2.rectangle(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size),
                          (int(pattern[4][2]), int(pattern[4][1]), int(pattern[4][0])), thickness=-1)

        # 8: blind
        if (pattern[8][0] == -1):
            transparent_hatch(startX + 3 * size, startY + 7 * size, startX + 4 * size, startY + 11 * size, transparent_hatch)
            transparent_hatch(startX + 3 * size, startY + 14 * size, startX + 4 * size, startY + 18 * size, transparent_hatch)
            transparent_hatch(startX + 7 * size, startY + 7 * size, startX + 8 * size, startY + 11 * size, transparent_hatch)
            transparent_hatch(startX + 7 * size, startY + 14 * size, startX + 8 * size, startY + 18 * size, transparent_hatch)
            transparent_hatch(startX + 11 * size, startY + 7 * size, startX + 12 * size, startY + 11 * size, transparent_hatch)
            transparent_hatch(startX + 15 * size, startY + 7 * size, startX + 16 * size, startY + 11 * size, transparent_hatch)


        elif (pattern[8][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 4 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 4 * size, startY + 18 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 12 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 15 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)

        # 5: cornice
        if (pattern[5][0] == -1):
            transparent_hatch(startX + 3 * size, startY + 7 * size, startX + 8 * size, startY + 8 * size, transparent_hatch)
            transparent_hatch(startX + 3 * size, startY + 14 * size, startX + 8 * size, startY + 15 * size, transparent_hatch)
            transparent_hatch(startX + 11 * size, startY + 7 * size, startX + 16 * size, startY + 8 * size, transparent_hatch)

        elif (pattern[5][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 8 * size, startY + 15 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)

        # 6: sill
        if (pattern[6][0] == -1):
            transparent_hatch(startX + 3 * size, startY + 10 * size, startX + 8 * size, startY + 11 * size, transparent_hatch)
            transparent_hatch(startX + 3 * size, startY + 17 * size, startX + 8 * size, startY + 18 * size, transparent_hatch)
            transparent_hatch(startX + 11 * size, startY + 10 * size, startX + 16 * size, startY + 11 * size, transparent_hatch)

        elif (pattern[6][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 3 * size, startY + 17 * size), (startX + 8 * size, startY + 18 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)

        # 7: balcony
        if (pattern[7][0] == -1):
            transparent_hatch(startX + 10 * size, startY + 10 * size, startX + 17 * size, startY + 12 * size, transparent_hatch)

        elif (pattern[7][0] != -1):
            cv2.rectangle(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size),
                          (int(pattern[7][2]), int(pattern[7][1]), int(pattern[7][0])), thickness=-1)

        # 9: deco
        if (pattern[9][0] == -1):
            transparent_hatch(startX + 1 * size, startY + 5 * size, startX + 18 * size, startY + 6 * size, transparent_hatch)
            transparent_hatch(startX + 1 * size, startY + 12 * size, startX + 18 * size, startY + 13 * size, transparent_hatch)

        elif (pattern[9][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 18 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 1 * size, startY + 12 * size), (startX + 18 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)

        # 10: molding
        if (pattern[10][0] == -1):
            transparent_hatch(startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 5 * size, transparent_hatch)

        elif (pattern[10][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size),
                          (int(pattern[10][2]), int(pattern[10][1]), int(pattern[10][0])), thickness=-1)

        # 11: pillar
        if (pattern[11][0] == -1):
            transparent_hatch(startX + 1 * size, startY + 5 * size, startX + 2 * size, startY + 19 * size, transparent_hatch)
            transparent_hatch(startX + 9 * size, startY + 5 * size, startX + 10 * size, startY + 19 * size, transparent_hatch)
            transparent_hatch(startX + 17 * size, startY + 5 * size, startX + 18 * size, startY + 19 * size, transparent_hatch)

        elif (pattern[11][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)

        # 12: shop
        if (pattern[12][0] == -1):
            transparent_hatch(startX + 2 * size, startY + 13 * size, startX + 9 * size, startY + 19 * size, transparent_hatch)

        elif (pattern[12][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size),
                          (int(pattern[12][2]), int(pattern[12][1]), int(pattern[12][0])), thickness=-1)

    cv2.addWeighted(transparent_hatch, alpha, canvas, 1 - alpha, 0, canvas)
    print("##############################################################################")

    cv2.imwrite(os.path.join(palette_folder, 'icons_random25_hatch.jpg'), canvas)
    print("Archichrome created")

# not using transparent_hatch function using for sentences instead, avoid overlap
def drawIcon_emptyShade_for(patterns, palette_folder):
    # draw canvas
    # iconMatrixNum = math.ceil(math.sqrt(len(patterns)))

    iconMatrixNum = math.floor(math.sqrt(len(patterns)))
    canvasSize = iconMatrixNum * iconSize + (iconMatrixNum + 1) * margin
    mod = math.pow(iconMatrixNum, 2)-len(patterns)
    if(mod < 0):
        canvasSizeH = (iconMatrixNum + 1)* iconSize + (iconMatrixNum + 2) * margin
    else:
        canvasSizeH = (iconMatrixNum)* iconSize + (iconMatrixNum + 1) * margin

    canvas = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)
    transparent_hatch = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)

    # print("elementsColor: ", elementsColor)

    # draw each icons
    for p, pattern in enumerate(patterns):
        # get the position of patterns[p]
        if (p == 0):
            iconRowNum = (p + 1) // iconMatrixNum
        else:
            iconRowNum = p // iconMatrixNum
        iconColumnNum = p % iconMatrixNum

        # calc start point(startX, startY) of icon No.[i]
        startX = (iconColumnNum + 1) * margin + iconColumnNum * iconSize
        startY = (iconRowNum + 1) * margin + iconRowNum * iconSize


        # 1: background
        if (pattern[1][0] == -1):
            hatch_pattern = [[startX, startY, startX + 19 * size, startY + 1 * size],
                             [startX, startY + 1 * size, startX + 1 * size, startY + 19 * size],
                             [startX + 18 * size, startY + 1 * size, startX + 19 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[1][0] != -1):
            cv2.rectangle(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)

        # 2: facade
        if (pattern[2][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(canvas, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(canvas, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[2][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[2][2]), int(pattern[2][1]), int(pattern[2][0])), thickness=-1)

        # 3: window
        if (pattern[3][0] == -1):
            hatch_pattern = [[startX + 4 * size, startY + 8 * size, startX + 7 * size, startY + 10 * size],
                             [startX + 12 * size, startY + 8 * size, startX + 15 * size, startY + 10 * size]]

            x0 = hatch[0]
            y0 = hatch[1]
            x1 = hatch[2]
            y1 = hatch[3]

            shadeNumX = int((x1 - x0) / shadeSize)
            shadeNumY = int((y1 - y0) / shadeSize)

            for r in range(shadeNumY):
                if (r % 2 == 0):
                    for c in range(math.ceil(shadeNumX / 2)):
                        cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                      (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                      (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                elif (r % 2 == 1):
                    for c in range(shadeNumX // 2):
                        cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                      (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                      (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                else:
                    pass



        elif (pattern[3][0] != -1):
            cv2.rectangle(canvas, (startX + 4 * size, startY + 8 * size), (startX + 7 * size, startY + 10 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 12 * size, startY + 8 * size), (startX + 15 * size, startY + 10 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)

        # 4: door
        if (pattern[4][0] == -1):
            hatch_pattern = [[startX + 12 * size, startY + 14 * size, startX + 15 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass



        elif (pattern[4][0] != -1):
            cv2.rectangle(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size),
                          (int(pattern[4][2]), int(pattern[4][1]), int(pattern[4][0])), thickness=-1)

        # 8: blind
        if (pattern[8][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 8 * size, startX + 4 * size, startY + 10 * size],
                             [startX + 7 * size, startY + 8 * size, startX + 8 * size, startY + 10 * size],
                             [startX + 11 * size, startY + 8 * size, startX + 12 * size, startY + 10 * size],
                             [startX + 15 * size, startY + 8 * size, startX + 16 * size, startY + 10 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[8][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 8 * size), (startX + 4 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 8 * size), (startX + 8 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 8 * size), (startX + 12 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 15 * size, startY + 8 * size), (startX + 16 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)

        # 5: cornice
        if (pattern[5][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 7 * size, startX + 8 * size, startY + 8 * size],
                             [startX + 11 * size, startY + 7 * size, startX + 16 * size, startY + 8 * size],]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[5][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)

        # 6: sill
        if (pattern[6][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 10 * size, startX + 8 * size, startY + 11 * size],
                             [startX + 11 * size, startY + 10 * size, startX + 16 * size, startY + 11 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[6][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)

        # 7: balcony
        if (pattern[7][0] == -1):
            hatch_pattern = [[startX + 10 * size, startY + 10 * size, startX + 17 * size, startY + 12 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[7][0] != -1):
            cv2.rectangle(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size),
                          (int(pattern[7][2]), int(pattern[7][1]), int(pattern[7][0])), thickness=-1)

        # 9: deco
        if (pattern[9][0] == -1):
            hatch_pattern = [[startX + 2 * size, startY + 5 * size, startX + 9 * size, startY + 6 * size],
                             [startX + 10 * size, startY + 5 * size, startX + 17 * size, startY + 6 * size],
                             [startX + 2 * size, startY + 12 * size, startX + 9 * size, startY + 13 * size],
                             [startX + 10 * size, startY + 12 * size, startX + 17 * size, startY + 13 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[9][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 5 * size), (startX + 9 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 10 * size, startY + 5 * size), (startX + 17 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 2 * size, startY + 12 * size), (startX + 9 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 10 * size, startY + 12 * size), (startX + 17 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)

        # 10: molding
        if (pattern[10][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 5 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[10][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size),
                          (int(pattern[10][2]), int(pattern[10][1]), int(pattern[10][0])), thickness=-1)

        # 11: pillar
        if (pattern[11][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 5 * size, startX + 2 * size, startY + 19 * size],
                             [startX + 9 * size, startY + 5 * size, startX + 10 * size, startY + 19 * size],
                             [startX + 17 * size, startY + 5 * size, startX + 18 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[11][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)

        # 12: shop
        if (pattern[12][0] == -1):
            hatch_pattern = [[startX + 2 * size, startY + 13 * size, startX + 9 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[12][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size),
                          (int(pattern[12][2]), int(pattern[12][1]), int(pattern[12][0])), thickness=-1)

    cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_canvas.jpg'), canvas)
    canvas_overlayed = cv2.addWeighted(transparent_hatch, alpha, canvas, 1-alpha, 0, canvas)
    print("##############################################################################")
    print("palette_folder: ", palette_folder)
    print("os.path.join(palette_folder)", os.path.join(palette_folder))
    cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_transparent_hatch.jpg'), transparent_hatch)
    cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_canvas_overlayed.jpg'), canvas_overlayed)
    print("Archichrome created")

def drawIcon_emptyShade_over(patterns, palette_folder):
    # draw canvas
    # iconMatrixNum = math.ceil(math.sqrt(len(patterns)))

    iconMatrixNum = math.floor(math.sqrt(len(patterns)))
    canvasSize = iconMatrixNum * iconSize + (iconMatrixNum + 1) * margin
    mod = math.pow(iconMatrixNum, 2)-len(patterns)
    if(mod < 0):
        canvasSizeH = (iconMatrixNum + 1)* iconSize + (iconMatrixNum + 2) * margin
    else:
        canvasSizeH = (iconMatrixNum)* iconSize + (iconMatrixNum + 1) * margin

    canvas = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)
    transparent_hatch = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)

    # print("elementsColor: ", elementsColor)

    # draw each icons
    for p, pattern in enumerate(patterns):
        # get the position of patterns[p]
        if (p == 0):
            iconRowNum = (p + 1) // iconMatrixNum
        else:
            iconRowNum = p // iconMatrixNum
        iconColumnNum = p % iconMatrixNum

        # calc start point(startX, startY) of icon No.[i]
        startX = (iconColumnNum + 1) * margin + iconColumnNum * iconSize
        startY = (iconRowNum + 1) * margin + iconRowNum * iconSize


        # 1: background
        if (pattern[1][0] == -1):
            hatch_pattern = [[startX, startY, startX + 19 * size, startY + 1 * size],
                             [startX, startY + 1 * size, startX + 1 * size, startY + 19 * size],
                             [startX + 18 * size, startY + 1 * size, startX + 19 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[1][0] != -1):
            cv2.rectangle(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)

        # 2: facade
        if (pattern[2][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(canvas, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(canvas, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[2][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[2][2]), int(pattern[2][1]), int(pattern[2][0])), thickness=-1)

        # 3: window
        if (pattern[3][0] == -1):
            hatch_pattern = [[startX + 4 * size, startY + 8 * size, startX + 7 * size, startY + 10 * size],
                             [startX + 12 * size, startY + 8 * size, startX + 15 * size, startY + 10 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(canvas, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(canvas, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass



        elif (pattern[3][0] != -1):
            cv2.rectangle(canvas, (startX + 4 * size, startY + 8 * size), (startX + 7 * size, startY + 10 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 12 * size, startY + 8 * size), (startX + 15 * size, startY + 10 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)

        # 4: door
        if (pattern[4][0] == -1):
            hatch_pattern = [[startX + 12 * size, startY + 14 * size, startX + 15 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass



        elif (pattern[4][0] != -1):
            cv2.rectangle(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size),
                          (int(pattern[4][2]), int(pattern[4][1]), int(pattern[4][0])), thickness=-1)

        # 8: blind
        if (pattern[8][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 8 * size, startX + 4 * size, startY + 10 * size],
                             [startX + 7 * size, startY + 8 * size, startX + 8 * size, startY + 10 * size],
                             [startX + 11 * size, startY + 8 * size, startX + 12 * size, startY + 10 * size],
                             [startX + 15 * size, startY + 8 * size, startX + 16 * size, startY + 10 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[8][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 8 * size), (startX + 4 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 8 * size), (startX + 8 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 8 * size), (startX + 12 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 15 * size, startY + 8 * size), (startX + 16 * size, startY + 10 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)

        # 5: cornice
        if (pattern[5][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 7 * size, startX + 8 * size, startY + 8 * size],
                             [startX + 11 * size, startY + 7 * size, startX + 16 * size, startY + 8 * size],]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[5][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)

        # 6: sill
        if (pattern[6][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 10 * size, startX + 8 * size, startY + 11 * size],
                             [startX + 11 * size, startY + 10 * size, startX + 16 * size, startY + 11 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[6][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)

        # 7: balcony
        if (pattern[7][0] == -1):
            hatch_pattern = [[startX + 10 * size, startY + 10 * size, startX + 17 * size, startY + 12 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[7][0] != -1):
            cv2.rectangle(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size),
                          (int(pattern[7][2]), int(pattern[7][1]), int(pattern[7][0])), thickness=-1)

        # 9: deco
        if (pattern[9][0] == -1):
            hatch_pattern = [[startX + 2 * size, startY + 5 * size, startX + 9 * size, startY + 6 * size],
                             [startX + 10 * size, startY + 5 * size, startX + 17 * size, startY + 6 * size],
                             [startX + 2 * size, startY + 12 * size, startX + 9 * size, startY + 13 * size],
                             [startX + 10 * size, startY + 12 * size, startX + 17 * size, startY + 13 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[9][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 5 * size), (startX + 9 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 10 * size, startY + 5 * size), (startX + 17 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 2 * size, startY + 12 * size), (startX + 9 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 10 * size, startY + 12 * size), (startX + 17 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)

        # 10: molding
        if (pattern[10][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 5 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[10][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size),
                          (int(pattern[10][2]), int(pattern[10][1]), int(pattern[10][0])), thickness=-1)

        # 11: pillar
        if (pattern[11][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 5 * size, startX + 2 * size, startY + 19 * size],
                             [startX + 9 * size, startY + 5 * size, startX + 10 * size, startY + 19 * size],
                             [startX + 17 * size, startY + 5 * size, startX + 18 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[11][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)

        # 12: shop
        if (pattern[12][0] == -1):
            hatch_pattern = [[startX + 2 * size, startY + 13 * size, startX + 9 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil((shadeNumX) / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize,
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)

                    elif (r % 2 == 1):
                        for c in range((shadeNumX) // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * shadeSize + shadeSize),
                                                   y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize),
                                           y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[12][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size),
                          (int(pattern[12][2]), int(pattern[12][1]), int(pattern[12][0])), thickness=-1)

    # canvas_overlayed = cv2.addWeighted(transparent_hatch, alpha, canvas, 1-alpha, 0, canvas)
    print("##############################################################################")
    print("palette_folder: ", palette_folder)
    print("os.path.join(palette_folder)", os.path.join(palette_folder))
    cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_canvas.jpg'), canvas)
    # cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_transparent_hatch.jpg'), transparent_hatch)
    # cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_canvas_overlayed.jpg'), canvas_overlayed)
    print("Archichrome created")

# not using transparent_hatch function
def drawIcon_emptyShade_white(patterns, palette_folder):
    # draw canvas
    # iconMatrixNum = math.ceil(math.sqrt(len(patterns)))

    iconMatrixNum = math.floor(math.sqrt(len(patterns)))
    canvasSize = iconMatrixNum * iconSize + (iconMatrixNum + 1) * margin
    mod = math.pow(iconMatrixNum, 2)-len(patterns)
    if(mod < 0):
        canvasSizeH = (iconMatrixNum + 1)* iconSize + (iconMatrixNum + 2) * margin
    else:
        canvasSizeH = (iconMatrixNum)* iconSize + (iconMatrixNum + 1) * margin

    canvas = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)
    transparent_hatch = np.full((canvasSizeH, canvasSize, 3), 255, dtype=np.uint8)

    # print("elementsColor: ", elementsColor)

    # draw each icons
    for p, pattern in enumerate(patterns):
        # get the position of patterns[p]
        if (p == 0):
            iconRowNum = (p + 1) // iconMatrixNum
        else:
            iconRowNum = p // iconMatrixNum
        iconColumnNum = p % iconMatrixNum

        # calc start point(startX, startY) of icon No.[i]
        startX = (iconColumnNum + 1) * margin + iconColumnNum * iconSize
        startY = (iconRowNum + 1) * margin + iconRowNum * iconSize


        # 2: facade
        if (pattern[2][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(canvas, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(canvas, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[2][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[2][2]), int(pattern[2][1]), int(pattern[2][0])), thickness=-1)

        # 1: background
        if (pattern[1][0] == -1):
            hatch_pattern = [[startX, startY, startX + 19 * size, startY + 1 * size],
                             [startX, startY + 1 * size, startX + 1 * size, startY + 19 * size],
                             [startX + 18 * size, startY + 1 * size, startX + 19 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        # 3: window
        elif (pattern[1][0] != -1):
            cv2.rectangle(canvas, (startX, startY), (startX + 19 * size, startY + 1 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX, startY + 1 * size), (startX + 1 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 18 * size, startY + 1 * size), (startX + 19 * size, startY + 19 * size),
                          (int(pattern[1][2]), int(pattern[1][1]), int(pattern[1][0])), thickness=-1)

        if (pattern[3][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 7 * size, startX + 8 * size, startY + 11 * size],
                             [startX + 11 * size, startY + 7 * size, startX + 16 * size, startY + 11 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass



        elif (pattern[3][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[3][2]), int(pattern[3][1]), int(pattern[3][0])), thickness=-1)

        # 4: door
        if (pattern[4][0] == -1):
            hatch_pattern = [[startX + 12 * size, startY + 14 * size, startX + 15 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[4][0] != -1):
            cv2.rectangle(canvas, (startX + 12 * size, startY + 14 * size), (startX + 15 * size, startY + 19 * size),
                          (int(pattern[4][2]), int(pattern[4][1]), int(pattern[4][0])), thickness=-1)

        # 8: blind
        if (pattern[8][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 7 * size, startX + 4 * size, startY + 11 * size],
                             [startX + 7 * size, startY + 7 * size, startX + 8 * size, startY + 11 * size],
                             [startX + 11 * size, startY + 7 * size, startX + 12 * size, startY + 11 * size],
                             [startX + 15 * size, startY + 7 * size, startX + 16 * size, startY + 11 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[8][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 4 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            # cv2.rectangle(canvas, (startX + 3 * size, startY + 14 * size), (startX + 4 * size, startY + 18 * size),
            #               (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 7 * size, startY + 7 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            # cv2.rectangle(canvas, (startX + 7 * size, startY + 14 * size), (startX + 8 * size, startY + 18 * size),
            #               (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 12 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 15 * size, startY + 7 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[8][2]), int(pattern[8][1]), int(pattern[8][0])), thickness=-1)

        # 5: cornice
        if (pattern[5][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 7 * size, startX + 8 * size, startY + 8 * size],
                             [startX + 11 * size, startY + 7 * size, startX + 16 * size, startY + 8 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[5][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 7 * size), (startX + 8 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 7 * size), (startX + 16 * size, startY + 8 * size),
                          (int(pattern[5][2]), int(pattern[5][1]), int(pattern[5][0])), thickness=-1)

        # 6: sill
        if (pattern[6][0] == -1):
            hatch_pattern = [[startX + 3 * size, startY + 10 * size, startX + 8 * size, startY + 11 * size],
                             [startX + 11 * size, startY + 10 * size, startX + 16 * size, startY + 11 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[6][0] != -1):
            cv2.rectangle(canvas, (startX + 3 * size, startY + 10 * size), (startX + 8 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 11 * size, startY + 10 * size), (startX + 16 * size, startY + 11 * size),
                          (int(pattern[6][2]), int(pattern[6][1]), int(pattern[6][0])), thickness=-1)

        # 7: balcony
        if (pattern[7][0] == -1):
            hatch_pattern = [[startX + 10 * size, startY + 10 * size, startX + 17 * size, startY + 12 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[7][0] != -1):
            cv2.rectangle(canvas, (startX + 10 * size, startY + 10 * size), (startX + 17 * size, startY + 12 * size),
                          (int(pattern[7][2]), int(pattern[7][1]), int(pattern[7][0])), thickness=-1)

        # 9: deco
        if (pattern[9][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 5 * size, startX + 18 * size, startY + 6 * size],
                             [startX + 1 * size, startY + 12 * size, startX + 18 * size, startY + 13 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[9][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 18 * size, startY + 6 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 1 * size, startY + 12 * size), (startX + 18 * size, startY + 13 * size),
                          (int(pattern[9][2]), int(pattern[9][1]), int(pattern[9][0])), thickness=-1)

        # 10: molding
        if (pattern[10][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 1 * size, startX + 18 * size, startY + 5 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass


        elif (pattern[10][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 1 * size), (startX + 18 * size, startY + 5 * size),
                          (int(pattern[10][2]), int(pattern[10][1]), int(pattern[10][0])), thickness=-1)

        # 11: pillar
        if (pattern[11][0] == -1):
            hatch_pattern = [[startX + 1 * size, startY + 5 * size, startX + 2 * size, startY + 19 * size],
                             [startX + 9 * size, startY + 5 * size, startX + 10 * size, startY + 19 * size],
                             [startX + 17 * size, startY + 5 * size, startX + 18 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[11][0] != -1):
            cv2.rectangle(canvas, (startX + 1 * size, startY + 5 * size), (startX + 2 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 9 * size, startY + 5 * size), (startX + 10 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)
            cv2.rectangle(canvas, (startX + 17 * size, startY + 5 * size), (startX + 18 * size, startY + 19 * size),
                          (int(pattern[11][2]), int(pattern[11][1]), int(pattern[11][0])), thickness=-1)

        # 12: shop
        if (pattern[12][0] == -1):
            hatch_pattern = [[startX + 2 * size, startY + 13 * size, startX + 9 * size, startY + 19 * size]]
            for hatch in hatch_pattern:
                x0 = hatch[0]
                y0 = hatch[1]
                x1 = hatch[2]
                y1 = hatch[3]

                shadeNumX = int((x1 - x0)/ shadeSize)
                shadeNumY = int((y1 - y0)/ shadeSize)

                for r in range(shadeNumY):
                    if (r % 2 == 0):
                        for c in range(math.ceil(shadeNumX / 2)):
                            cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    elif (r % 2 == 1):
                        for c in range(shadeNumX // 2):
                            cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + shadeSize), y0 + r * shadeSize),
                                          (x0 + (2 * c * shadeSize + 2 * shadeSize), y0 + (r * shadeSize + shadeSize)),
                                          (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
                    else:
                        pass

        elif (pattern[12][0] != -1):
            cv2.rectangle(canvas, (startX + 2 * size, startY + 13 * size), (startX + 9 * size, startY + 19 * size),
                          (int(pattern[12][2]), int(pattern[12][1]), int(pattern[12][0])), thickness=-1)

    cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_canvas_white.jpg'), canvas)
    canvas_overlayed = cv2.addWeighted(transparent_hatch, alpha, canvas, 1-alpha, 0)
    print("##############################################################################")
    print("palette_folder: ", palette_folder)
    print("os.path.join(palette_folder)", os.path.join(palette_folder))
    cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_transparent_hatch_white.jpg'), transparent_hatch)
    cv2.imwrite(os.path.join(palette_folder, 'icons_randam_hatch_canvas_overlayed_white.jpg'), canvas_overlayed)
    print("Archichrome created")


def transparent_hatch(x0, y0, x1, y1, transparent_hatch):
    # draw transparent_hatch pattern
    shadeNumX = int((x1 - x0) / shadeSize)
    shadeNumY = int((y1 - y0) / shadeSize)
    for r in range(shadeNumY):
        if (r % 2 == 0):
            for c in range(math.ceil(shadeNumX / 2)):
                cv2.rectangle(transparent_hatch, (x0 + 2 * c * shadeSize, y0 + r * size), (x1 + (2 * c * size + size), y1 + (r * size + size)), (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
        elif (r % 2 == 1):
            for c in range(shadeNumX // 2):
                cv2.rectangle(transparent_hatch, (x0 + (2 * c * size + size), y0 + r * size), (x1 + (2 * c * size + 2 * size), y1 + (r * size + size)), (shadeColor[2], shadeColor[1], shadeColor[0]), thickness=-1)
        else:
            pass

# check if elementColor contains elements class in structure elementsColor[][]
# this version is coded for elementsColor[]

def select_maxminNum(elementsColor, elements_len):

    sort_color = []
    sort_len = sorted(elements_len, reverse=True)
    # count = 0
    # indexList = []
    # usedIndex = []
    for i in range(len(sort_len)):
        if elements_len.count(sort_len[i]) == 1:
            sort_color.append(elementsColor[elements_len.index(elements_len[i])])
        elif elements_len.count(sort_len[i]) > 1 and elements_len.count(sort_len[i]) == count - 1:
            count = 0
            usedIndex = []
            indexList = [j for j, x in enumerate(elements_len) if x == sort_len[i]]
            if indexList[count] not in usedIndex:
                sort_color.append(elementsColor[indexList[count]])
                usedIndex.append(indexList[count])
                count += 1
            else:
                pass
        elif elements_len.count(sort_len[i]) > 1:
            indexList = [j for j, x in enumerate(elements_len) if x == sort_len[i]]
            if indexList[count] not in usedIndex:
                sort_color.append(elementsColor[indexList[count]])
                usedIndex.append(indexList[count])
                count += 1
            else:
                pass

        else:
            pass

    max_color = sort_color[0]
    min_color = sort_color[len(sort_color)-1]
    elementColor = [max_color, min_color]

    return elementColor

def select_maxminDist(elementsColor, elements_len):
    dist_array = []
    for i,col0 in enumerate(elementsColor):
        dist = 0
        for j, col1 in enumerate(elementsColor):
            lab0 = clustering.rgb2lab(col0)
            lab1= clustering.rgb2lab(col1)
            dist += clustering.cie2000(lab0, lab1)
        dist_array.append(dist)
    max_dist = max(dist_array)
    min_dist = min(dist_array)
    return dist


############
if __name__ == '__main__':

    print("##############################################################################")


    icons = drawIcon(elementsColor, elementsNames)
