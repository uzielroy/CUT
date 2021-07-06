import glob
import random
import os
import glob
import zipfile
import io
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils import images_pca
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



class ImageDataset_kaggle(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(make_dataset_kaggle(root,False))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(io.BytesIO(self.files_A[index % len(self.files_A)])).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(io.BytesIO(self.files_B[random.randint(0, len(self.files_B) - 1)])).convert('RGB'))
        else:
            item_B = self.transform(Image.open(io.BytesIO(self.files_B[index % len(self.files_B)])).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def make_dataset_kaggle(path,monet=False):
    train_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    }
    extension = "tfrec.zip"
    if monet:
        for item in os.listdir(os.path.join(path,'monet'):
            if item.endswith('tfrec.zip') and item.startswith('monet'):
                file_name = os.path.abspath(item)
                zip_ref = zipfile.ZipFile(file_name)
                zip_ref.extractall(os.path.join(path,'monet'))
                zip_ref.close() # close file

        train_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        train_files = glob.glob(os.path.join(path,'monet/*.tfrec'))
        train_ids = []
        train_class = []
        train_images = []
        for i in train_files:
          train_image_dataset = tf.data.TFRecordDataset(i)
          train_image_dataset = train_image_dataset.map(_parse_image_function)
          images = [image_features['image'].numpy() for image_features in train_image_dataset]
          train_images = train_images + images
          train_images = images_pca(images)
    else:
        for item in os.listdir(os.path.join(path,'photo/*.tfrec')): # loop through items in dir
            if item.endswith('tfrec.zip') and item.startswith('photo'): # check for "148.tfrec.zip" extension
                file_name = os.path.abspath(item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall(os.path.join(path,'photo/*.tfrec')) # extract file to dir
                zip_ref.close() # close file

        train_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        train_files = glob.glob(os.path.join(path,'photo/*.tfrec'))
        train_ids = []
        train_class = []
        train_images = []
        for i in train_files:
          train_image_dataset = tf.data.TFRecordDataset(i)
          train_image_dataset = train_image_dataset.map(_parse_image_function)
          images = [image_features['image'].numpy() for image_features in train_image_dataset]
          train_images = train_images + images

    return train_images


def _parse_image_function(example_proto):
 train_feature_description = {'image': tf.io.FixedLenFeature([], tf.string),
 }
 return tf.io.parse_single_example(example_proto, train_feature_description)
