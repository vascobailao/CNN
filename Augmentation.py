import cv2
import numpy as np
from math import ceil, floor

class Augmentation:

    def __init__(self):
        return 0


    def get_translate_parameters(self, index):
        if index == 0:  # Translate left 20 percent
            offset = np.array([0.0, 0.2], dtype=np.float32)
            size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
            w_start = 0
            w_end = int(ceil(0.8 * IMAGE_SIZE))
            h_start = 0
            h_end = IMAGE_SIZE
        elif index == 1:  # Translate right 20 percent
            offset = np.array([0.0, -0.2], dtype=np.float32)
            size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
            w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
            w_end = IMAGE_SIZE
            h_start = 0
            h_end = IMAGE_SIZE
        elif index == 2:  # Translate top 20 percent
            offset = np.array([0.2, 0.0], dtype=np.float32)
            size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
            w_start = 0
            w_end = IMAGE_SIZE
            h_start = 0
            h_end = int(ceil(0.8 * IMAGE_SIZE))
        else:  # Translate bottom 20 percent
            offset = np.array([-0.2, 0.0], dtype=np.float32)
            size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
            w_start = 0
            w_end = IMAGE_SIZE
            h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
            h_end = IMAGE_SIZE

        return offset, size, w_start, w_end, h_start, h_end

    def get_mask_coord(self, imshape):
        vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                              (0.43 * imshape[1], 0.32 * imshape[0]),
                              (0.56 * imshape[1], 0.32 * imshape[0]),
                              (0.85 * imshape[1], 0.99 * imshape[0])]], dtype=np.int32)
        return vertices

    def get_perspective_matrices(self, X_img):
        offset = 15
        img_size = (X_img.shape[1], X_img.shape[0])

        # Estimate the coordinates of object of interest inside the image.
        src = np.float32(get_mask_coord(X_img.shape))
        dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                          [img_size[0] - offset, img_size[1]]])

        perspective_matrix = cv2.getPerspectiveTransform(src, dst)
        return perspective_matrix

    def perspective_transform(self, X_img):
        # Doing only for one type of example
        perspective_matrix = get_perspective_matrices(X_img)
        warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                         (X_img.shape[1], X_img.shape[0]),
                                         flags=cv2.INTER_LINEAR)
        return warped_img

    def add_gaussian_noise(self, X_imgs):
        gaussian_noise_imgs = []
        row, col, _ = X_imgs[0].shape
        # Gaussian distribution parameters
        mean = 0
        var = 0.1
        sigma = var ** 0.5

        for X_img in X_imgs:
            gaussian = np.random.random((row, col, 1)).astype(np.float32)
            gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
            gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
            gaussian_noise_imgs.append(gaussian_img)
        gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
        return gaussian_noise_imgs


    def add_salt_pepper_noiseadd_salt(self, X_imgs):
        # Need to produce a copy as to not modify the original image
        X_imgs_copy = X_imgs.copy()
        row, col, _ = X_imgs_copy[0].shape
        salt_vs_pepper = 0.2
        amount = 0.004
        num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

        for X_img in X_imgs_copy:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
            X_img[coords[0], coords[1], :] = 1

            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
            X_img[coords[0], coords[1], :] = 0
        return X_imgs_copy

