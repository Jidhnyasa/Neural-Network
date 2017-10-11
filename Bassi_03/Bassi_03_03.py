# Bassi, Aman
# 1001-393-217
# 2017-10-09
# Assignment_03_03

from __future__ import division
import numpy as np


# reading data from the file and then normalizing it
def reading_data(file_name):
    # Each row of data in the file becomes a row in the matrix
    # So the resulting matrix has dimension [num_samples x sample_dimension]
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
    max_price, max_volume = max(data[:, 0]), max(data[:, 1])
    # print(max_price, max_volume)
    # print(data, data.shape)
    data[:, 0], data[:, 1] = (data[:, 0]/max_price)-0.5, (data[:, 1]/max_volume)-0.5
    # print(data)
    # print(data[1, 0])

    return data


