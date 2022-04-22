import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imageio import imread
from imageio import imwrite
import random


def read_images(root_path):
    pictures = np.array([])

    # iterate over all files and folders
    for dirpath, dirnames, files in os.walk(root_path):
        if files:
            # filter out non jpg files
            jpg_only = [jpg for jpg in files if 'jpg' in jpg]
            # choose one random image from each folder
            choose = random.choice(jpg_only)
            # read the chosen image
            face = imread(os.path.join(dirpath, choose)).reshape(-1, 1)

            # stack all images in one array
            if pictures.size == 0:
                pictures = face
            else:
                pictures = np.column_stack((pictures, face))

    return pictures


def write_images(image_mat, dir_name):
    os.makedirs(dir_name)

    for i in range(image_mat.shape[1]):
        file_name = f"{dir_name}\\face_{i}.jpg"
        imwrite(file_name, image_mat[:, i].reshape((200, 180, 3)))


def plot_images(images, n_row, n_col, shape):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[:, i].reshape(shape), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def norm_matrix(mat):
    # return a normalized matrix
    return (mat - np.min(mat, axis=0)) / (np.max(mat, axis=0) - np.min(mat, axis=0))


if __name__ == "__main__":
    
    # get data
    root_path = os.path.join(os.getcwd(), 'faces94')
    images = read_images(root_path)
    plot_images(images, 10, 10, (200, 180, 3))
    
    # normalize image data
    norm_original = norm_matrix(images)
    
    # calculate covariance matrix
    cov_mat = (norm_original.T @ norm_original) / (images.shape[0] - 1)
    
    # get eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    
    # sort principle components according to influence:
    sort_ind = np.argsort(eig_val)[::-1]
    sorted_eig_val = eig_val[sort_ind]
    sorted_eig_vec = eig_vec[:, sort_ind]
    
    # explained variance ratio
    norm_eig_vals = sorted_eig_val / sorted_eig_val.sum()
    
    # find the first n components that explain 95% of the variance
    cum_eig_vals = np.cumsum(norm_eig_vals)
    bool_components = cum_eig_vals <= 0.95
    reduced_eig_vecs = sorted_eig_vec[:, bool_components]
    
    # project images onto PCA space:
    projected_images = norm_original @ reduced_eig_vecs

    # normalize to RGB data and plot eigenfaces:
    norm_proj_images = np.round(255 * norm_matrix(projected_images))
    plot_images(norm_proj_images.astype(np.uint8), 5, 5, (200, 180, 3))
    
    # write images to file:
    write_images(norm_proj_images.astype(np.uint8), 'results')
