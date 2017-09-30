# Bassi, Aman
# 1001-393-217
# 2017-09-27
# Assignment_02_03

import numpy as np
import scipy.misc
import os
import random

# function for getting images input of 784*1000 which means each image is an array of 784 values and there are 1000 images , also getting the images actual values in a list
def read_one_image_and_convert_to_vector(folder_name):
    img_list = np.zeros((784, 1000))
    image_number = 0
    image_digit = []

    for file_name in os.listdir(folder_name):
        # getting the first number from the image
        string_image = (os.path.basename(file_name))
        a = int(string_image[0])
        image_digit.append(a)

        # reading image one by one and comverting 28 * 28 into a vector and then equating it with the columns of our img_list
        img = scipy.misc.imread(os.path.join(folder_name, file_name)).astype(np.float32)
        img = img.reshape(-1)
        # print(img.shape)
        img_list[:, image_number] = img
        # print(temp.shape)
        image_number = image_number + 1
        #print(img_list[:,0])
    # print(img_list.shape)
    # print(len(image_digit), image_digit)
    return img_list, image_digit

# a, b=read_one_image_and_convert_to_vector("mnist_images")
# print(a.shape)
# this is a function for getting 80 percent training data and 20 percent testing data randomly from the image data ie the input image data
def getting_data(images_input):
    indices = random.sample(range(0, 1000), 1000) # generating random unique indices from 0 to 1000 and generating 1000 values
    training_data = images_input[:, indices[0:800]] # so taking the first 800 indices and getting images according to those 800 indices
    test_data = images_input[:, indices[800:1000]] # taking last 200 indices and getting the columns or images according to those 200 indices so basically getting 200 images for those columns
    # print(training_data.shape, test_data.shape)
    return training_data, test_data, indices