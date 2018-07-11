import numpy as np
import h5py


class HDF5:

    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:

            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and labels from the HDF dataset

                images = self.db["images"][i: i + self.batchSize]

                labels = self.db["labels"][i: i + self.batchSize]