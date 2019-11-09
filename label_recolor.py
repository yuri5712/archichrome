import os
import cv2
import datetime


class Recolor:
    def __init__(self, folder="CMP_facade_DB_base"):
        self.target_path = os.listdir(os.path.join(folder, 'targets_original'))
        self.target = self.recolor_target(os.path.join(folder), self.target_path)
        self.input_path = os.listdir(os.path.join(folder, 'inputs_original'))
        self.target = self.recolor_input(os.path.join(folder), self.input_path)

    def recolor_target(self, folder, train_files_target):
        for i,file_target in enumerate(train_files_target):
            if (file_target[-4:] == '.png'):

                target_image = os.path.join(folder, 'targets_original', file_target)
                target_image = cv2.imread(target_image, 1)  # load color 3channels>0, grayscale=0, image itself<0

                for r,row in enumerate(target_image):
                    for c,column in enumerate(row):

                        # nolabel
                        if column[0] == 0 and column[1] == 0 and column[2] == 0:
                            target_image[r][c] = [0, 0, 0]

                        # background
                        elif column[0] == 170 and column[1] == 0 and column[2] == 0:
                            target_image[r][c] = [85, 0, 0]

                        # facade
                        elif column[0] == 255 and column[1] == 0 and column[2] == 0:
                            target_image[r][c] = [170, 0, 0]

                        # window
                        elif column[0] == 255 and column[1] == 85 and column[2] == 0:
                            target_image[r][c] = [255, 85, 0]

                        # door
                        elif column[0] == 255 and column[1] == 170 and column[2] == 0:
                            target_image[r][c] = [255, 170, 0]

                        # cornice
                        elif column[0] == 255 and column[1] == 255 and column[2] == 0:
                            target_image[r][c] = [255, 255, 85]

                        # sill
                        elif column[0] == 170 and column[1] == 255 and column[2] == 85:
                            target_image[r][c] = [255, 255, 170]

                        # blind
                        elif column[0] == 85 and column[1] == 255 and column[2] == 170:
                            target_image[r][c] = [255, 170, 255]
                            # print ("blind", column)

                        # balcony
                        elif column[0] == 0 and column[1] == 255 and column[2] == 255:
                            target_image[r][c] = [255, 85, 255]
                            # print ("balcony", column)


                        # deco
                        elif column[0] == 0 and column[1] == 170 and column[2] == 255:
                            target_image[r][c] = [170, 0, 255]
                            # print ("deco", column)


                        # molding
                        elif column[0] == 0 and column[1] == 85 and column[2] == 255:
                            target_image[r][c] = [85, 0, 255]
                            # print ("molding", column)


                        # pillar
                        elif column[0] == 0 and column[1] == 0 and column[2] == 255:
                            target_image[r][c] = [0, 0, 170]
                            # print ("pillar", column)


                        # shop
                        elif column[0] == 0 and column[1] == 0 and column[2] == 170:
                            target_image[r][c] = [0, 0, 85]
                            # print ("shop", column)


                        else:
                            pass


                        # for p,pix in enumerate(column):
                        #
                        #     # if pix == 0:
                        #     #     target_image[r][c][p] = 31.875
                        #     # elif pix == 85:
                        #     #     target_image[r][c][p] = 95.625
                        #     # elif pix == 170:
                        #     #     target_image[r][c][p] = 159.375
                        #     # elif pix == 255:
                        #     #     target_image[r][c][p] = 223.125
                        #     else:
                        #         pass
                cv2.imwrite(os.path.join(os.path.join(folder, 'targets'), file_target[:-4] + '.png'), target_image)

                print("file no.", i, " : ", file_target)
            else:
                pass

    def recolor_input(self, folder,      train_files_target):
        for i,file_target in enumerate(train_files_target):
            if (file_target[-4:] == '.jpg'):

                target_image = os.path.join(folder, 'inputs_original', file_target)
                target_image = cv2.imread(target_image, 1)  # load color 3channels>0, grayscale=0, image itself<0

                for r,row in enumerate(target_image):
                    for c,column in enumerate(row):
                        for p,pix in enumerate(column):
                            target_image[r][c][p] = pix
                cv2.imwrite(os.path.join(os.path.join(folder, 'inputs'), file_target[:-4] + '.jpg'), target_image)

                print("file no.", i, " : ", file_target)
            else:
                pass



if __name__ == '__main__':
    start = datetime.datetime.today()
    print('start time:', start)
    print("##############################################################################")

    recolor = Recolor()

    print("##############################################################################")
    end = datetime.datetime.today()
    print('end time:', end)