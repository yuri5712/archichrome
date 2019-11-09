import cv2
import sys
import os
from math import ceil
import numpy as np

if __name__ == "__main__":

    folder = '.\CMP_facade_DB_base\base\inputs'

    train_files, validation_files, test_files = train_valid_test_split(os.listdir(os.path.join(folder, 'inputs')))
    train_inputs, train_targets = file_paths_to_images(folder, train_files)
    test_inputs, test_targets = file_paths_to_images(folder, test_files, True)

    

    def resize_images(self, folder):
        # Create 250*250 Size Image in new dir
        # original train input = otinpur, resized train input = rtinput

        #resized_train_input = []
        #resized_train_target = []

        size = (250, 250)
        for otinput, ottarget in train_inputs, train_targets:
            resized_train_input.append(rtinput=cv2.resize(otinput, size))
            resized_train_target.append(rttarget=cv2.resize(ottarget, size))

            cv2.imwrite(os.path.join('.\resized_CMP\inputs', 'input' + i + '.jpg'), hogeImage)
            cv2.imwrite(os.path.join('.\resized_CMP\targets', 'target' + i + '.jpg'), hogeImage)
            i = i + 1

        return resized_train_input, resized_train_target


    def file_paths_to_images(self, folder, files_list, verbose=False):
        inputs = []
        targets = []

        for file in files_list:
            input_image = os.path.join(folder, 'inputs', file)
            target_image = os.path.join(folder, 'targets' if include_hair else 'targets_face_only', file)

            test_image = np.array(cv2.imread(input_image, 0))  # load grayscale
            # test_image = np.multiply(test_image, 1.0 / 255)
            inputs.append(test_image)

            target_image = cv2.imread(target_image, 0)
            target_image = cv2.threshold(target_image, 127, 1, cv2.THRESH_BINARY)[1]#target image pixcels assigned to black or white
            targets.append(target_image)

        return inputs, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, 0.15, 0.15)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )