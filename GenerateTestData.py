# the function of this code is to create
# dataset for testing from the existing
# pictures from the training dataset
# pictures are randomly picked

import os
import random

TRAIN_DIR = 'train dataset/'
TEST_DIR = 'test data/'

folders = os.listdir(TRAIN_DIR)
for folder in folders:
    if not os.path.exists(TEST_DIR + folder + '/'):
        os.makedirs(TEST_DIR + folder + '/')
    files = os.listdir(TRAIN_DIR + folder + '/')
    random_list = random.sample(range(0, len(files)), k=6)
    for i in range(0, len(random_list)):
        print('value of random int: ', random_list[i])
        filename = files[random_list[i]]
        os.replace(TRAIN_DIR + folder + '/' + filename, TEST_DIR + folder + '/' + filename)
        print(folder, '/', filename, 'moved to test data folder')

