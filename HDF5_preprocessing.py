from random import shuffle
import glob
import numpy as np
import h5py
shuffle_data = True

class HDF5_preprocessing:

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.hdf5_dir = base_dir+"/dataset.hdf5"
        self.train_addrs = None
        self.train_labels = None
        self.val_addrs = None
        self.val_labels = None
        self.test_addrs = None
        self.test_labels = None
        self.shuffle_data = True

    def create_table(self):

        addrs_train = glob.glob(self.base_dir+"/train/*.jpg")
        addrs_test = glob.glob(self.base_dir+"/test/*.jpg")

        labels_train = [0 if 'cat' in addr else 1 for addr in addrs_train]
        labels_test = [0 if 'cat' in addr else 1 for addr in addrs_train]

        if shuffle_data:

            c = list(zip(addrs_train, labels_train))
            d = list(zip(addrs_test, labels_test))
            shuffle(c)
            shuffle(d)
            addrs_train, labels_train = zip(*c)
            addrs_test, labels_test = zip(*d)

        self.train_addrs = addrs_train[0:int(0.8 * len(addrs_train))]
        self.train_labels = labels_train[0:int(0.8 * len(labels_train))]

        self.val_addrs = addrs_train[int(0.8 * len(addrs_train)):]
        self.val_labels = labels_train[int(0.8 * len(addrs_train)):]

        self.test_addrs = addrs_test[int(len(addrs_test)):]
        self.test_labels = labels_test[int(len(labels_test)):]


    def create_hdf5(self):

        train_shape = (len(self.train_addrs), 224, 224, 3)
        val_shape = (len(self.val_addrs), 224, 224, 3)
        test_shape = (len(self.test_addrs), 224, 224, 3)

        hdf5_file = h5py.File(self.hdf5_dir, mode='w')
        hdf5_file.create_dataset("train_img", train_shape, np.int8)
        hdf5_file.create_dataset("val_img", val_shape, np.int8)
        hdf5_file.create_dataset("test_img", test_shape, np.int8)
        hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
        hdf5_file.create_dataset("train_labels", (len(self.train_addrs),), np.int8)
        hdf5_file["train_labels"][...] = self.train_labels
        hdf5_file.create_dataset("val_labels", (len(self.val_addrs),), np.int8)
        hdf5_file["val_labels"][...] = self.val_labels
        hdf5_file.create_dataset("test_labels", (len(self.test_addrs),), np.int8)
        hdf5_file["test_labels"][...] = self.test_labels




















