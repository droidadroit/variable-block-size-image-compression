# VARIABLE BLOCK-SIZE COMPRESSION BY HORIZONTAL MERGING

import numpy as np
from scipy import misc, spatial
from math import log, exp, pow
from sklearn.metrics import mean_squared_error
import cv2
from math import sqrt
from scipy.cluster.vq import vq, kmeans, whiten
from math import log
import scipy.misc
from bitarray import bitarray
import sys


def mse(image_a, image_b):
    # calculate mean square error between two images
    err = np.sum((image_a.astype(float) - image_b.astype(float)) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err


def psnr(a):

    tmp = (255*255)/float(a)
    return 10*log(tmp, 10)


def replicate(array, x):

    global image_height, image_width
    array = np.repeat(array, x, axis=1)
    return array


def count_zeros(array, ind, limit):

    x = 0
    while True:
        if array[ind] == 0:
            x += 1
            ind += 1
        else:
            break
        if ind % limit == 0:
            break
    return x


def get_centroids(x, p):
    # double the centroids after each iteration
    final_centroids = np.copy(x)
    for centroid in x:
        final_centroids = np.vstack((final_centroids, np.add(centroid, p)))
    return final_centroids


def num_pixels(vector, pixel_val, prob, const):

    x = 0
    for val in vector:
        if val == pixel_val:
            x += 1
    return x/float(vector.size) > const * prob


def unite(array, x):

    global block_height
    final = []

    for _i in range(0, array.shape[1], x):
        _j = 0
        total = np.zeros(block_height)
        while _j < x:
            total = np.sum([array[:, _i + _j], total], axis=0)
            # print total
            _j += 1
        total = [p / float(x) for p in total]
        final.append(total)
    return final

# source image
image_location = sys.argv[1]
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
image_height = len(image)
image_width = len(image[0])

threshold_1 = int(sys.argv[6])
threshold_2 = int(sys.argv[7])
canny_image = cv2.Canny(image, threshold_1, threshold_2)      # image after canny edge detection

number_of_blacks, number_of_whites = (0, 0)

for row in canny_image:
    for pixel in row:
        if pixel == 0:
            number_of_blacks += 1
        else:
            number_of_whites += 1

edge_prob = number_of_whites/float(image_height*image_width)
non_edge_prob = 1 - edge_prob

# dimensions of an image block
block_width = int(sys.argv[3])
block_height = int(sys.argv[4])
vector_dimension = block_width*block_height

bits_per_codevector = int(sys.argv[2])
codebook_size = pow(2, bits_per_codevector)
perturbation_vector = np.full(vector_dimension, 10)
mu = float(sys.argv[5])

count = 0
block_bit_array = bitarray(image_height*image_width/vector_dimension)
block_index = 0
image_vectors = []*image_height

# MERGING
for index_i, i in enumerate(range(0, image_height, block_height)):
    count = 0
    non_edge_array = []
    for index_j, j in enumerate(range(0, image_width, block_width)):
        vec = np.reshape(canny_image[i:i+block_width, j:j+block_height], vector_dimension)
        is_edge_block = num_pixels(vec, 255, edge_prob, mu)
        if is_edge_block:
            if count != 0:
                united = np.transpose(unite(non_edge_array, count))
                image_vectors.append(np.reshape(united, vector_dimension))
                count = 0
            block_bit_array[block_index] = 1
            vec = np.reshape(image[i:i + block_width, j:j + block_height], vector_dimension)
            image_vectors.append(vec)
        else:
            block_bit_array[block_index] = 0
            count += 1
            if count == 1:
                non_edge_array = image[i:i + block_width, j:j + block_height]
            else:
                non_edge_array = np.hstack([non_edge_array, image[i:i + block_width, j:j + block_height]])
        block_index += 1
    if count != 0:
        united = np.transpose(unite(non_edge_array, count))
        image_vectors.append(np.reshape(united, vector_dimension))

image_vectors = np.asarray(image_vectors).astype(float)
number_of_image_vectors = image_vectors.shape[0]
actual_image_vectors = image_width*image_height/(block_width*block_height)

centroid_vector = np.mean(image_vectors, axis=0)
centroids = np.vstack((centroid_vector, np.add(centroid_vector, perturbation_vector)))
whitened = whiten(np.asarray(image_vectors))
reconstruction_values, distortion = kmeans(image_vectors, centroids)

for i in range(0, int(log(codebook_size/2, 2)), 1):
    reconstruction_values = get_centroids(reconstruction_values, perturbation_vector)
    reconstruction_values, distortion = kmeans(image_vectors, reconstruction_values)

image_vector_indices, distance = vq(image_vectors, reconstruction_values)

image_after_compression = np.zeros([image_width, image_height], dtype="uint8")

n = 0
index = 0

#  REPLICATION
while n < actual_image_vectors:
    if block_bit_array[n] == 0:
        c = count_zeros(block_bit_array, n, image_width/block_width)
    else:
        c = 1
    rv = reconstruction_values[image_vector_indices[index]]
    start_row = int(n / (image_width / block_width)) * block_height
    end_row = start_row + block_height
    start_column = (n * block_width) % image_width
    end_column = start_column + c*block_width
    d = replicate(np.reshape(rv, (block_width, block_height)), c)
    image_after_compression[start_row:end_row, start_column:end_column] = d
    index += 1
    n += c

output_image_name = "CB_size=" + str(codebook_size) + ".png"
scipy.misc.imsave(output_image_name, image_after_compression)

mean_squared_error = mse(image, image_after_compression)

print "Mean squared error = ", mean_squared_error
print "PSNR = ", psnr(mean_squared_error)
