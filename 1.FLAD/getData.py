import numpy as np
import gzip
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

'''
GetDataSet:mnist
    train_data:(60000,784)
    train_label:(60000,10), one-hot
    test_data:(10000,784)
    test_label:(10000,10), one-hot
    np.ndarray
'''

'''
GetDataSet:cifar-10
    train_data:(50000,3,32,32)
    train_label:(50000,), 
    test_data:(10000,3,32,32)
    test_label:(10000,),
    tensor 
'''

class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name =='cifar_10':
            self.cifarDataSetConstruct(isIID)
            
            
    def cifarDataSetConstruct(self,isIID):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # First fill 0 all around, then crop the image randomly to 32*32
                transforms.RandomHorizontalFlip(),  # Half the probability that the image is flipped, half the probability that it is not flipped
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Mean and variance used for normalization for each layer of R,G,B
                ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_data = torchvision.datasets.CIFAR10("../data/CIFAR_10",train=True,transform=transform_train,download=True)
        test_data = torchvision.datasets.CIFAR10("../data/CIFAR_10",train=False,transform=transform_test,download=True)
        
     
        self.train_data_size = len(train_data)
        self.test_data_size = len(test_data)
        train_dataloader = DataLoader(dataset=train_data, batch_size = self.train_data_size)
        test_dataloader = DataLoader(dataset=test_data, batch_size = self.test_data_size)
        
        for data in train_dataloader:
            train_images, train_labels = data
        for data in test_dataloader:
            test_images, test_labels = data
        
        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else: # Sort trains by tag size
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
            
        order = np.arange(self.test_data_size)
        np.random.shuffle(order)        
        self.test_data = test_images[order]
        self.test_label = test_labels[order]
        

    def mnistDataSetConstruct(self, isIID):
        data_dir = r'../data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = (train_images/255.0-0.1307)/0.3081
        test_images = test_images.astype(np.float32)
        test_images = (test_images/255.0-0.1307)/0.3081

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
            
        order = np.arange(self.test_data_size)
        np.random.shuffle(order)        
        self.test_data = test_images[order]
        self.test_label = test_labels[order]



def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet('mnist', False) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
    
    
    
    

