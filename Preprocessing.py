import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d

class Preprocessing:

    def __init__(self, rMean, gMean, bMean, width, height, horiz, inter):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def mean_preprocessing(self, image):

        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        return cv2.merge([B, G, R])

    def patch_preprocessing(self, image):

        return extract_patches_2d(image, (self.height, self.width),max_patches=1)[0]

    def crop_preprocessing(self, image):

        crops = []
        (h, w) = image.shape[:2]

        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]]

        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation = self.inter)
            crops.append(crop)

        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
            return np.array(crops)








