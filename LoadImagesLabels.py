# main purpose is to preprocess both train and test images
# to be ready to be fed to the knn model
# functions include loading the data from the passed source path
# as well as create txt files for the float training images and labels

import cv2
import imutils
import numpy as np
import os
# from PIL import Image


class ImgPreprocessing:

    flattened_images = []
    labels_ref = []
    labels_id = []
    final_labels = []
    string_labels = []
    new_width = 30
    new_height = 30
    min_contour_area = 700
    label_id = 1
    path = ''
    is_train_data = False
    file_path = None
    file_name = None
    img_size = None
    img_processed = False

    def load_data(self, path, is_train_data):
        self.path = path
        self.is_train_data = is_train_data
        self.flattened_images = np.empty((0, self.new_width * self.new_height))
        folders = os.listdir(self.path)

        # train and test images are inside folders with their respective longhand translation as folder name
        # iterate through folders
        for folder in folders:
            self.process_data(folder)

        # convert flattened images to float32 data type for knn feeding
        self.flattened_images = np.float32(self.flattened_images)

        # saving the images, labels IDs, and label reference as txt files
        if self.is_train_data:
            float_labels = np.array(self.labels_id, np.float32)
            self.final_labels = float_labels.reshape((float_labels.size, 1))
            np.savetxt('flattened_images.txt', self.flattened_images)
            np.savetxt('float_labels.txt', self.final_labels)
            np.savetxt('labels_ref.txt', self.labels_ref, delimiter=",", fmt="%s")
    # end of load_data()

    def get_path(self, folder_name, file_name):
        self.file_path = self.path + folder_name + "/" + file_name
        self.file_name = file_name
    # end of get_path()

    def process_data(self, folder):
        files = os.listdir(self.path + folder + "/")

        # iterate through all images in the folder
        for file_name in files:
            self.get_path(folder, file_name)
            read_img = cv2.imread(self.file_path)
            if self.is_train_data:
                print(self.file_path)
            height, width, channels = read_img.shape    # take img shape as reference for border detection
            self.img_size = width * height              # area of img calculated for border detection later
            self.preprocess_images(read_img)
            # if not self.is_train_data:
            #     self.preprocess_images(read_img)
            # else:
            #     # rotate image from -45 to 45 degrees
            #     i = -45
            #     while i <= 45:
            #         rotated = imutils.rotate_bound(read_img, i)
            #         self.preprocess_images(rotated)
            #         self.labels_id.append(float(self.label_id))
            #         i += 1
            if self.img_processed:
                self.string_labels.append(folder)

        if self.is_train_data:
            # append label ID and label name to the reference file
            self.labels_ref.append(float(self.label_id))
            self.labels_ref.append(folder)
            self.label_id += 1
    # end of process_train_data()

    def preprocess_images(self, read_img):
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

        img_thresh = closing.copy()
        # img_thresh_copy = img_thresh
        # # finding contours in the image
        # contours, heirarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        cropped_img = self.crop_img(img_thresh)

        # rects = self.crop_img(contours, img_thresh)

        # separate contours to another function and have its return value as rects
        # for contour in contours:
        #     col = []
        #     # checking if contour area is large enough to be considered a part of the stroke
        #     # or if contour area is as large as perimeter of image, hence the border
        #     if self.min_contour_area < cv2.contourArea(contour) < self.img_size * 0.8:
        #         [x, y, width, height] = cv2.boundingRect(contour)
        #         col.append(x)
        #         col.append(y)
        #         col.append(width)
        #         col.append(height)
        #         rects.append(col)
        #     # else, the contour area is considered as noise and drawn black, same as the background
        #     else:
        #         cv2.drawContours(img_thresh, contour, -1, (0, 0, 0), 3)

        # # finding location, width, and height of the stroke's bounding box for cropping
        # rects_len = range(len(rects))
        # try:
        #     x = min(rects[i][0] for i in rects_len)
        #     y = min(rects[i][1] for i in rects_len)
        #     w = max(rects[i][0] + rects[i][2] for i in rects_len) - x
        #     h = max(rects[i][1] + rects[i][3] for i in rects_len) - y
        # except:
        #     print('image', self.file_name,  'is blurry, cannot be processed...')
        #     self.img_processed = False
        #     return
        #
        # # this is here if ever we want to see which part of the image is the bounding box placed
        # # still need to add cv.imshow('label', src) tho
        # # attach bounding box to character
        # cv2.rectangle(read_img,         # draw rectangle on original training image
        #               (x, y),           # upper left corner
        #               (x + w, y + h),   # lower right corner
        #               (0, 0, 255),      # red
        #               2)
        #
        # cropped_img = img_thresh[y:y+h, x:x+w]

        # rotating img here instead of before calling preprocess images
        if self.is_train_data:
            i = -45
            while i <= 45:
                rotated = imutils.rotate_bound(cropped_img, i)
                cropped_img = self.crop_img(rotated)
                cropped_resized_img = cv2.resize(cropped_img, (self.new_width, self.new_height))  # must resize image to make all images uniform
                flat_img = cropped_resized_img.reshape((1, self.new_width * self.new_height))  # reshape array to 1D
                self.flattened_images = np.append(self.flattened_images, flat_img, 0)  # append image to the list of flattened images
                self.labels_id.append(float(self.label_id))
                i += 1
        else:
            cropped_resized_img = cv2.resize(cropped_img, (self.new_width, self.new_height)) # must resize image to make all images uniform
            flat_img = cropped_resized_img.reshape((1, self.new_width * self.new_height))  # reshape array to 1D
            self.flattened_images = np.append(self.flattened_images, flat_img, 0)  # append image to the list of flattened images

        self.img_processed = True
    # end process_images

    def crop_img(self, img_thresh):
        # finding contours in the image
        img_thresh_copy = img_thresh
        contours, heirarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        rects = []
        for contour in contours:
            col = []
            # checking if contour area is large enough to be considered a part of the stroke
            # or if contour area is as large as perimeter of image, hence the border
            if self.min_contour_area < cv2.contourArea(contour) < self.img_size * 0.8:
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
        try:
            x = min(rects[i][0] for i in rects_len)
            y = min(rects[i][1] for i in rects_len)
            w = max(rects[i][0] + rects[i][2] for i in rects_len) - x
            h = max(rects[i][1] + rects[i][3] for i in rects_len) - y
        except:
            print('image', self.file_name, 'is blurry, cannot be processed...')
            self.img_processed = False
            return

        cropped_img = img_thresh[y:y + h, x:x + w]
        return cropped_img

# end ImgPreprocessing