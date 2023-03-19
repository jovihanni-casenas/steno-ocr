# main purpose is to generate the txt files for the training data
# as well as the labels ID and its reference file

from LoadImagesLabels import *

TRAIN_DIR = 'train dataset/'
is_train_data = True

gen_data = ImgPreprocessing()
gen_data.load_data(TRAIN_DIR, True)