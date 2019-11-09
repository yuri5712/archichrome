import os
import cv2
import datetime
import infer
import icon
from main import Main
import numpy as np
import matplotlib
import shutil


class Difference:
    def __init__(self, folder="street_color"):
        folder = os.path.join(folder, 'street02')
        self.file_path = os.listdir(folder)
        print(self.file_path)

        # self.difference,self.ratio = self.color_difference_by_street(folder)
        # self.recolor_segmentation_DLimgs(folder)
        self.make_comparison(folder)

    # # old version
    # def color_difference(self, folder, train_files_target):
    #     color_difference = []
    #     space = 1
    #     # print("train_files_target", train_files_target)
    #     for i,file in enumerate(train_files_target):
    #         print("file:", file)
    #         file = os.path.join(folder, 'street02\paths11_right',  file)
    #         img_color = cv2.imread(file, 1)  # load color 3channels>0, grayscale=0, image itself<0
    #         # print(img_color)
    #         num = 0
    #         color = [0, 0, 0]
    #         for r,row in enumerate(img_color):
    #             for c,col in enumerate(row):
    #                 if space<r<(len(img_color)-space) and space<c<(len(row)-space):
    #                     color += col
    #                     num += 1
    #         print("num", num)
    #         r = int(color[2]/num)
    #         g = int(color[1]/num)
    #         b = int(color[0]/num)
    #         print([b, g, r])
    #         color_difference.append([b, g, r])
    #     print(color_difference)
    #     return color_difference

    # use this one

    def color_difference_by_street(self, folder):
        color_difference = []
        color_ratio = []
        space = 1 #clear spaces from edges
        # print("train_files_target", train_files_target)
        paths = os.listdir(folder)
        for p,path in enumerate(paths):
            # print(path)
            files_path = os.path.join(folder, path)
            files = os.listdir(files_path)
            color_difference.append([])
            color_ratio.append([])
            for s,section in enumerate(files):
                sections_path = os.path.join(files_path, section)
                sections = os.listdir(sections_path)
                # print("sections:", sections)

                # get mean color bgr
                color_difference[p].append([[],[],[],[]])
                for f,file in enumerate(sections):
                    # print("file:", file[11:-4:])
                    if file[-4:] == ".png": #-4
                        # print("file:", file)
                        file_path = os.path.join(sections_path, file)
                        # print("file_path: ", file_path)
                        img_color = cv2.imread(file_path, 1)  # load color 3channels>0, grayscale=0, image itself<0
                        # print("img_color", img_color)
                        num = 0
                        color = [0, 0, 0]
                        for r,row in enumerate(img_color):
                            for c,col in enumerate(row):
                                if space<r<(len(img_color)-space) and space<c<(len(row)-space):
                                    color += col
                                    num += 1
                        # print("num", num)
                        r = int(color[2]/num)
                        g = int(color[1]/num)
                        b = int(color[0]/num)
                        # print([b, g, r])
                        # print("file[:8]:", file[:8])
                        if file[:7]=="paths11":
                            if file[12:-4:] == "GoogleStreetView":
                                color_difference[p][s][0]=[b, g, r]
                            elif file[12:-4:] == "RealColor":
                                color_difference[p][s][1]=[b, g, r]
                            elif file[12:-4:] == "gear360":
                                color_difference[p][s][2]=[b, g, r]
                            elif file[12:-4:] == "ThetaS":
                                color_difference[p][s][3]=[b, g, r]
                            else:
                                print("warning!!!!!!!!!!!!!", sections_path, "/", file[11:-4:], "is not appropriate!!!!!!!!!!!!!!!!")
                        else:
                            if file[11:-4:] == "GoogleStreetView":
                                color_difference[p][s][0]=[b, g, r]
                            elif file[11:-4:] == "RealColor":
                                color_difference[p][s][1]=[b, g, r]
                            elif file[11:-4:] == "gear360":
                                color_difference[p][s][2]=[b, g, r]
                            elif file[11:-4:] == "ThetaS":
                                color_difference[p][s][3]=[b, g, r]
                            else:
                                print("warning!!!!!!!!!!!!!", sections_path, "/", file[11:-4:], "is not appropriate!!!!!!!!!!!!!!!!")

                # get color difference ratio
                color_ratio[p].append([])
                # print("color_ratio[p][s]", color_ratio[p][s])
                for i in range(len(color_difference[p][s])):
                    # print("color_difference[p][s][0][2]:", color_difference[p][s][0][2])
                    # print("color_difference[p][s][i][2]:", color_difference[p][s][i][2])
                    r = color_difference[p][s][0][2]/color_difference[p][s][i][2]
                    g = color_difference[p][s][0][1]/color_difference[p][s][i][1]
                    b = color_difference[p][s][0][0]/color_difference[p][s][i][0]
                    # r = color_difference[p][s][i][2]/color_difference[p][s][0][2]
                    # g = color_difference[p][s][i][1]/color_difference[p][s][0][1]
                    # b = color_difference[p][s][i][0]/color_difference[p][s][0][0]
                    ratioBGR = [b, g, r]
                    color_ratio[p][s].append(ratioBGR)
        print("ratio:", color_ratio)
        print("color_difference", color_difference)
        return color_difference, color_ratio

    def recolor_segmentation_DLimgs(self, folder):
        paths = os.listdir(folder)
        for p,path in enumerate(paths):
            print
            if path == "paths11_left":
                files_path = os.path.join(folder, path)
                # print("file_path:", files_path)
                files = os.listdir(files_path)
                for s,section in enumerate(files):
                    section_path = os.path.join(files_path, section)
                    DLimg_path = os.path.join(files_path, section, "DLimgs")
                    print("DLimg_path:", DLimg_path)
                    DLimg_files = os.listdir(DLimg_path)
                    # print("DLimg_files:", DLimg_files)

                    print("##############################################################################")
                    print("i am now working on:", section_path)

                    if len(DLimg_files) == 1:
                        original_path = os.path.join(DLimg_path, DLimg_files[0]) #[s]==streetviewXX.jpg
                    elif len(DLimg_files) == 4:
                        original_path = os.path.join(DLimg_path, DLimg_files[2]) #[s]==streetviewXX.jpg
                    else:
                        print("warning!!! num =! 1or4")
                        original_path = os.path.join(DLimg_path, DLimg_files[0])  # [s]==streetviewXX.jpg
                    DLimg = cv2.imread(original_path, 1)
                    # print("DLimg:", DLimg)
                    DLimg_original = DLimg
                    DL_recolor = DLimg
                    # print("DLimg_original:", DLimg_original)
                    print("DLimg before", DL_recolor[0][0])
                    # print("DLimg_original:", DLimg_original)
                    # create color changed DLimgs
                    for i in range(len(self.ratio[p][s])):
                        if i == 0:
                                key = "Google_streetview"
                                key_dir, dlimgs_dir, streetview_dir, segmented_dir, palette_dir = self.make_dir(files_path, section, key)
                                cv2.imwrite(os.path.join(dlimgs_dir, key + '.jpg'), DL_recolor)
                                label_color = self.segmentation_infer(key_dir)
                                elementsColor, usedElementsNames = Main.callCluster(palette_dir, label_color)
                                draw = icon.drawIcon(elementsColor, usedElementsNames, palette_dir)
                        else:
                            for r, row in enumerate(DLimg_original):
                                for c, column in enumerate(row):
                                    # print("column:", column)
                                    # print("self.ratio[p][s][i]:", self.ratio[p][s][i])
                                    for x, pix in enumerate(column):
                                        if pix ==0:
                                            DL_recolor[r][c][x] = pix
                                        else:
                                            # DL_recolor[r][c][p] = int(DLimg_original[r][c][p] * self.ratio[p][s][i][p])
                                            # print("DLimg_original[r][c][x]:", DLimg_original[r][c][x])
                                            # print("self.ratio[p][s][i][x]:", self.ratio[p][s][i][x])
                                            DL_recolor[r][c][x] = DLimg_original[r][c][x] * self.ratio[p][s][i][x]
                            print("DLimg after", DL_recolor[0][0])
                            DL_recolor = np.array(DL_recolor)
                            print("DL_recolor:", DL_recolor.dtype)
                            if i == 1:
                                key = "RealColor"
                                key_dir, dlimgs_dir, streetview_dir, segmented_dir, palette_dir = self.make_dir(files_path,
                                                                                                                section,
                                                                                                                key)
                                cv2.imwrite(os.path.join(dlimgs_dir, key + '.jpg'), DL_recolor)
                                label_color = self.segmentation_infer(key_dir)
                                elementsColor, usedElementsNames = Main.callCluster(palette_dir, label_color)
                                draw = icon.drawIcon(elementsColor, usedElementsNames, palette_dir)
                            elif i == 2:
                                key = "gear360"
                                key_dir, dlimgs_dir, streetview_dir, segmented_dir, palette_dir = self.make_dir(files_path,
                                                                                                                section,
                                                                                                                key)
                                cv2.imwrite(os.path.join(dlimgs_dir, key + '.jpg'), DL_recolor)
                                label_color = self.segmentation_infer(key_dir)
                                elementsColor, usedElementsNames = Main.callCluster(palette_dir, label_color)
                                draw = icon.drawIcon(elementsColor, usedElementsNames, palette_dir)
                            elif i == 3:
                                key = "ThetaS"
                                key_dir, dlimgs_dir, streetview_dir, segmented_dir, palette_dir = self.make_dir(files_path,
                                                                                                                section,
                                                                                                                key)
                                cv2.imwrite(os.path.join(dlimgs_dir, key + '.jpg'), DL_recolor)
                                label_color = self.segmentation_infer(key_dir)
                                elementsColor, usedElementsNames = Main.callCluster(palette_dir, label_color)
                                draw = icon.drawIcon(elementsColor, usedElementsNames, palette_dir)
                            else:
                                pass
            else:
                print("##############################################################################")
                print("this is ", os.path.join(folder, path), "skip segmentation")

    def segmentation_infer(self, dir):
        segmentation = infer.Test(folder=os.path.join(dir, "DLimgs"),
                                  resized_folder=os.path.join(dir, "streetview"),
                                  segmented_folder=os.path.join(dir, "segmented"))
        return segmentation.label_color

    def make_dir(self, files_path, section, key):
        key_dir = os.path.join(files_path, section, key)
        dlimgs_dir = os.path.join(key_dir, "DLimgs")
        streetview_dir = os.path.join(key_dir, "streetview")
        segmented_dir = os.path.join(key_dir, "segmented")
        palette_dir = os.path.join(key_dir, "palette")
        if not os.path.exists(key_dir):
            os.makedirs(key_dir)
            os.makedirs(dlimgs_dir)
            os.makedirs(streetview_dir)
            os.makedirs(segmented_dir)
            os.makedirs(palette_dir)
        else:
            pass

        return key_dir, dlimgs_dir, streetview_dir, segmented_dir, palette_dir

    def make_comparison(self, folder):
        paths = os.listdir(folder)
        for p, path in enumerate(paths):
            files_path = os.path.join(folder, path)
            files = os.listdir(files_path)
            for s, section in enumerate(files):
                print("section:", section)
                sections_path = os.path.join(files_path, section)
                sections = os.listdir(sections_path)
                print("##############################################################################")
                print("i am now working on:", sections_path)

                sample_dir = os.path.join(sections_path, "samples")
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                normal_path = os.path.join(files_path, section, "Google_streetview", "streetview")
                normal_files = os.listdir(normal_path)
                normal_img = os.path.join(normal_path, normal_files[0])

                comparison_dir = os.path.join(sections_path, "comparison")

                if os.path.exists(comparison_dir):
                    shutil.rmtree(comparison_dir)
                    print("comparison_dir DELETED")
                os.makedirs(comparison_dir)
                for f, file in enumerate(sections):
                    if file[-4:] == ".png":
                        if file[:7] == "paths11":
                            img_path = os.path.join(sections_path, file)
                            print("img_path:", img_path)
                            if file[12:-4:] == "GoogleStreetView":
                                key = "Google_streetview"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            elif file[12:-4:] == "RealColor":
                                key = "RealColor"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            elif file[12:-4:] == "gear360":
                                key = "gear360"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            elif file[12:-4:] == "ThetaS":
                                key = "ThetaS"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            else:
                                print("not working, flile name: ", file)
                        else:
                            img_path = os.path.join(sections_path, file)
                            print("img_path:", img_path)
                            if file[11:-4:] == "GoogleStreetView":
                                key = "Google_streetview"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            elif file[11:-4:] == "RealColor":
                                key = "RealColor"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            elif file[11:-4:] == "gear360":
                                key = "gear360"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            elif file[11:-4:] == "ThetaS":
                                key = "ThetaS"
                                self.sample_for_comparison(img_path, normal_img, sample_dir, key)
                                self.concat_horizon(files_path, section, key, comparison_dir)
                            else:
                                print("not working, flile name: ", file)
                    else:
                        print("this is not png, flile name: ", file)
                comparison_files = os.listdir(comparison_dir)
                if len(comparison_files) == 4:
                    for con_h in comparison_files:
                        print("os.path.join(comparison_dir, con_h): ", os.path.join(comparison_dir, con_h))
                        print("con_h[:-4]:", con_h[:-4])
                        if con_h[:-4] == "Google_streetview":
                            con_v0 = cv2.imread(os.path.join(comparison_dir, con_h), 1)
                            # print("con_v0:", con_v0)
                        elif con_h[:-4] == "RealColor":
                            con_v1 = cv2.imread(os.path.join(comparison_dir, con_h), 1)
                        elif con_h[:-4] == "gear360":
                            con_v2 = cv2.imread(os.path.join(comparison_dir, con_h), 1)
                        elif con_h[:-4] == "ThetaS":
                            con_v3 = cv2.imread(os.path.join(comparison_dir, con_h), 1)
                    imgs_v = cv2.vconcat([con_v0, con_v1, con_v2, con_v3])
                    cv2.imwrite(os.path.join(comparison_dir, "comparison.jpg"), imgs_v)

    def sample_for_comparison(self, sample_path, normal_path, sample_dir, sample_name):
        # print("sample_path:", sample_path)
        # print("normal_path:", normal_path)
        sample = cv2.imread(sample_path, 1)
        normal = cv2.imread(normal_path, 1)
        # print("sample_len:", len(sample))
        # print("sample_len[0]:", len(sample[0]))
        # print(type(sample))
        # print(sample)
        s_h,s_w = sample.shape[:2]
        n_h,n_w = normal.shape[:2]
        if(s_h != n_h or s_w != n_w):
            # background of the palette
            sample_resized = np.full((n_h, n_w, 3), 255, dtype=np.uint8)

            # top = int(n_h/2 - s_h/2)
            # left = int(n_w/2 - s_w/2)
            # normal[top:n_h + top:, left:n_w + left] = sample_resized

            sample_resized[int(n_h/2 - s_h/2):int(n_h/2 + s_h/2), int(n_w/2 - s_w/2):int(n_w/2 + s_w/2)] = sample

            # background = np.full((n_h, n_w, 3), 255, dtype=np.uint8)
            # sample_resized =self.cvpaste_center(sample, background)
            # print("sample_resized:", sample_resized)
            print(("sample_dir:", sample_dir))
            cv2.imwrite(os.path.join(sample_dir, sample_name+".jpg"), sample_resized)
        else:
            cv2.imwrite(os.path.join(sample_dir, sample_name), sample)

    def concat_horizon(self, files_path, section, key, comparison_dir):
        key_dir = os.path.join(files_path, section, key)
        dlimgs_dir = os.path.join(key_dir, "DLimgs")
        streetview_dir = os.path.join(key_dir, "streetview")
        segmented_dir = os.path.join(key_dir, "segmented")
        palette_dir = os.path.join(key_dir, "palette")

        color_sample = os.path.join(files_path, section, "samples", key+".jpg")
        dlimgs_file = os.listdir(dlimgs_dir)
        streetview_file = os.listdir(streetview_dir)
        segmented_file = os.listdir(segmented_dir)
        palette_file = os.listdir(palette_dir)

        # print("dlimgs_file:", dlimgs_file)
        print("os.path.join(key_dir, dlimgs_file[0]):", os.path.join(key_dir, dlimgs_file[0]))
        color_sample_img = cv2.imread(color_sample, 1)
        dlimgs_img = cv2.imread(os.path.join(dlimgs_dir, dlimgs_file[0]), 1)
        streetview_file = cv2.imread(os.path.join(streetview_dir, streetview_file[0]), 1)
        segmented_file_0 = cv2.imread(os.path.join(segmented_dir, segmented_file[0]), 1)
        segmented_file_1 = cv2.imread(os.path.join(segmented_dir, segmented_file[1]), 1)

        # palette_file = cv2.imread(palette_file, 1)

        # print("color_sample_img:", len(color_sample_img))
        # print("dlimgs_img:", len(dlimgs_img))
        # print("streetview_file:", len(streetview_file))
        # print("segmented_file_0:", len(segmented_file_0))
        # print("segmented_file_1:", len(segmented_file_1))

        h_imgs = [color_sample_img, streetview_file, segmented_file_0, segmented_file_1]
        # for num,h_img in enumerate(h_imgs):
        #     print("this is no.", num, "len(h_img):", len(h_img), "len(h_img[0])", len(h_img[0]))
        imgs_h = cv2.hconcat(h_imgs) # no palettes
        cv2.imwrite(os.path.join(comparison_dir, key+".jpg"), imgs_h)

    def cvpaste_center(self, img, imgback):
        # x and y are the distance from the center of the background image

        r = img.shape[0]
        c = img.shape[1]
        rb = imgback.shape[0]
        cb = imgback.shape[1]
        hrb = round(rb / 2)
        hcb = round(cb / 2)
        hr = round(r / 2)
        hc = round(c / 2)

        # Copy the forward image and move to the center of the background image
        imgrot = np.zeros((rb, cb, 3), np.uint8)
        imgrot[hrb - hr:hrb + hr, hcb - hc:hcb + hc, :] = img[:hr * 2, :hc * 2, :]

        # Makeing mask
        imggray = cv2.cvtColor(imgrot, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of the forward image in the background image
        img1_bg = cv2.bitwise_and(imgback, imgback, mask=mask_inv)

        # Take only region of the forward image.
        img2_fg = cv2.bitwise_and(imgrot, imgrot, mask=mask)

        # Paste the forward image on the background image
        imgpaste = cv2.add(img1_bg, img2_fg)

        return imgpaste










if __name__ == '__main__':
    start = datetime.datetime.today()
    print('start time:', start)
    print("##############################################################################")

    difference = Difference()

    print("##############################################################################")
    end = datetime.datetime.today()
    print('end time:', end)