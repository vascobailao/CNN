from random import shuffle
import glob
import numpy as np
import h5py
shuffle_data = True

class Preparation:

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















