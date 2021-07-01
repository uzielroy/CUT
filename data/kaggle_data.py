"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import tensorflow as tf

from PIL import Image
import os
import os.path
import glob
import zipfile


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _parse_image_function(example_proto):
 return tf.io.parse_single_example(example_proto, train_feature_description)

def make_dataset_kaggle(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def make_dataset_kaggle(monet=False):
    train_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    }
    extension = "tfrec.zip"
    if monet:
        for item in os.listdir('/content/CUT/kaggle_dataset/'): # loop through items in dir
            if item.endswith('tfrec.zip') and item.startswith('monet'): # check for "148.tfrec.zip" extension
                file_name = os.path.abspath(item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall('/content/CUT/kaggle_dataset/monet') # extract file to dir
                zip_ref.close() # close file

        train_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        train_files = glob.glob('/content/CUT/kaggle_dataset/monet/*.tfrec')
        train_ids = []
        train_class = []
        train_images = []
        for i in train_files:
          train_image_dataset = tf.data.TFRecordDataset(i)
          train_image_dataset = train_image_dataset.map(_parse_image_function)
          images = [image_features['image'].numpy() for image_features in train_image_dataset]
          train_images = train_images + images
    else:
        for item in os.listdir('/content/CUT/kaggle_dataset/'): # loop through items in dir
            if item.endswith('tfrec.zip') and item.startswith('photo'): # check for "148.tfrec.zip" extension
                file_name = os.path.abspath(item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall('/content/CUT/kaggle_dataset/photo') # extract file to dir
                zip_ref.close() # close file

        train_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        train_files = glob.glob('/content/CUT/kaggle_dataset/photo/*.tfrec')
        train_ids = []
        train_class = []
        train_images = []
        for i in train_files:
          train_image_dataset = tf.data.TFRecordDataset(i)
          train_image_dataset = train_image_dataset.map(_parse_image_function)
          images = [image_features['image'].numpy() for image_features in train_image_dataset]
          train_images = train_images + images

    return train_files




def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
