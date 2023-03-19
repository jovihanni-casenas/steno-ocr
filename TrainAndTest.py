# main purpose is to train and test the data

import numpy as np
from LoadImagesLabels import *
import os
import cv2
from PIL import Image

TRAIN_IMGS = 'flattened_images.txt'
TRAIN_LABELS = 'float_labels.txt'
LABEL_REF = 'labels_ref.txt'
TEST_DIR = 'test data/'
is_train_data = False


class TrainAndTest:
    train_images = None
    train_labels = []
    label_ref = []
    test_images = ImgPreprocessing()
    knn = cv2.ml.KNearest_create()
    pred_label = None

    def load_train_data(self):
        # loading images and labels from the txt files
        self.train_images = np.loadtxt(TRAIN_IMGS, np.float32)
        self.train_labels = np.loadtxt(TRAIN_LABELS, np.float32)
        data_ref = open(LABEL_REF, "r").read()
        self.label_ref = data_ref.split("\n")

    def process_test_images(self, path):
        # loads data using the member function of the ImgPreprocessing class
        self.test_images.load_data(path, is_train_data)

    def train_model(self):
        self.knn.train(self.train_images, cv2.ml.ROW_SAMPLE, self.train_labels)

    def find_nearest_neighbor(self):
        float_test_images = self.test_images.flattened_images

        for test_img in float_test_images:
            test_img = test_img.reshape((1, 900))
            retval, results, neigh_resp, dists = self.knn.findNearest(test_img, k=5)
            self.find_pred_label(results[0][0])
            # commented out, might be used later for viewing the test img
            # test_img = np.reshape(test_img, (30, 30))
            # im = Image.fromarray(test_img)
            # im = im.convert('RGB')
            # im.show(self.pred_label)

    def find_pred_label(self, float_label):
        # iterates through the labels reference file, stops when the label IDs match
        for i in range(len(self.label_ref)):
            if str(self.label_ref[i]) == str(float_label):
                self.pred_label = self.label_ref[i+1]
                print(self.pred_label)
                break


# =============================================================================

def main():
    knn_data = TrainAndTest()
    knn_data.load_train_data()
    print('done loading train data...')
    knn_data.process_test_images(TEST_DIR)
    print('done processing test images...')
    knn_data.train_model()
    knn_data.find_nearest_neighbor()
    print('done training model...')


if __name__ == '__main__':
    main()
