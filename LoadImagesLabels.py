# main purpose is to preprocess both train and test images
# to be ready to be fed to the knn model
# functions include loading the data from the passed source path
# as well as create txt files for the float training images and labels

import cv2
import numpy as np
import os
# from PIL import Image


class ImgPreprocessing:

    flattened_images = []
    labels_ref = []
    labels_id = []
    final_labels = []
    new_width = 30
    new_height = 30
    min_contour_area = 80
    label_id = 1
    path = ''
    is_train_data = False

    def load_data(self, path, is_train_data):
        self.path = path
        self.is_train_data = is_train_data
        self.flattened_images = np.empty((0, self.new_width * self.new_height))
        folders = os.listdir(self.path)

        # train images are inside folders with their respective longhand translation as folder name
        # iterate through folders
        for folder in folders:
            # generate list of file names if this function is called for training data
            if self.is_train_data:
                files = os.listdir(self.path + folder + "/")
                # iterate through all images in the folder
                for file_name in files:
                    self.preprocess_images(folder, file_name)
                    self.labels_id.append(float(self.label_id))
                # append label ID and label name to the reference file
                self.labels_ref.append(float(self.label_id))
                self.labels_ref.append(folder)
                self.label_id += 1
            # considers the 'folder' name as the file name for testing data as this category of images are placed in a single folder 'test data'
            else:
                file_name = folder
                self.preprocess_images("", file_name)

        # convert flattened images to float32 data type for knn feeding
        self.flattened_images = np.float32(self.flattened_images)

        # saving the images, labels IDs, and label reference as txt files
        if is_train_data:
            float_labels = np.array(self.labels_id, np.float32)
            self.final_labels = float_labels.reshape((float_labels.size, 1))
            np.savetxt('flattened_images.txt', self.flattened_images)
            np.savetxt('float_labels.txt', self.final_labels)
            np.savetxt('labels_ref.txt', self.labels_ref, delimiter=",", fmt="%s")
    # end of load_data()

    def preprocess_images(self, folder_name, file_name):
        # file path has additional folder in it if training data is to be processed
        if self.is_train_data:
            file_path = self.path + folder_name + "/" + file_name
        else:
            file_path = self.path + file_name

        read_img = cv2.imread(file_path)                                    # reading image
        img_gray = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)               # converting it to grayscale
        img_blurred = cv2.GaussianBlur(img_gray, (9, 9), 0)                 # blurring to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))       # getting kernel for erosion
        img_thresh = cv2.adaptiveThreshold(img_blurred,                     # input image
                                  255,                                      # make pixels that pass the threshold full white
                                  cv2.ADAPTIVE_THRESH_MEAN_C,               # use gaussian rather than mean, seems to give better results
                                  cv2.THRESH_BINARY_INV,                    # invert so foreground will be white, background will be black
                                  11,                                       # size of a pixel neighborhood used to calculate threshold value
                                  2)
        erosion = cv2.erode(img_thresh, kernel, iterations=1)               # remove small white pixels / noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))          # getting new kernel for dilation
        dilation = cv2.dilate(erosion, kernel, iterations=1)                # edges of strokes might be eroded, thus, dilated again slightly
        kernel_for_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_for_close) # some areas w/in the strokes are open, this is to make sure the strokes are solid

        img_thresh = dilation.copy()
        img_thresh_copy = img_thresh
        # finding contours in the image
        contours, heirarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        rects = []
        for contour in contours:
            col = []
            # checking if contour area is large enough to be considered a part of the stroke
            if cv2.contourArea(contour) > self.min_contour_area:
                [x, y, width, height] = cv2.boundingRect(contour)
                col.append(x)
                col.append(y)
                col.append(width)
                col.append(height)
                rects.append(col)
            # else, the contour area is considered as noise and drawn black, same as the background
            else:
                cv2.drawContours(img_thresh, contour, -1, (0, 0, 0), 3)

        # finding location, width, and height of the stroke's bounding box for cropping
        rects_len = range(len(rects))
        x = min(rects[i][0] for i in rects_len)
        y = min(rects[i][1] for i in rects_len)
        w = max(rects[i][0] + rects[i][2] for i in rects_len) - x
        h = max(rects[i][1] + rects[i][3] for i in rects_len) - y

        # this is here if ever we want to see which part of the image is the bounding box placed
        # still need to add cv.imshow('label', src) tho
        # attach bounding box to character
        cv2.rectangle(read_img,         # draw rectangle on original training image
                      (x, y),           # upper left corner
                      (x + w, y + h),   # lower right corner
                      (0, 0, 255),      # red
                      2)

        cropped_img = img_thresh[y:y+h, x:x+w]
        cropped_resized_img = cv2.resize(cropped_img, (self.new_width, self.new_height)) # must resize image to make all images uniform
        flat_img = cropped_resized_img.reshape((1, self.new_width * self.new_height))    # reshape array to 1D
        self.flattened_images = np.append(self.flattened_images, flat_img, 0)            # append image to the list of flattened images
    # end load_images

# end ImgPreprocessing