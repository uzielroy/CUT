import random
import time
import datetime
import sys
import os
from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from PIL import Image
import io
from torch.utils.tensorboard import SummaryWriter

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.writer = SummaryWriter()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log_board(self,losses,images):
        for i, loss_name in enumerate(losses.keys()):
            self.writer.add_scalar('Loss/'+str(loss_name), losses[loss_name].item(), self.batch+(self.batches_epoch*(self.epoch-1)))
        if self.batch%100==0:
            for image_name, tensor in images.items():
                self.writer.add_image(image_name,tensor2image(tensor.data),self.batch+(self.batches_epoch*(self.epoch-1)))
        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            self.epoch += 1
            self.batch = 1
        else:
            self.batch +=1
    def log(self, n_iter, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1



class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def graipher(pts, K):
    idx_pts = np.zeros(K)
    farthest_pts = np.zeros((K, 2))
    rand_idx = np.random.randint(len(pts))
    farthest_pts[0] = pts[rand_idx]
    idx_pts[0]=rand_idx
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        max_idx = np.argmax(distances)
        farthest_pts[i] = pts[max_idx]
        idx_pts[i] = max_idx
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, idx_pts.astype(int)


def images_pca(images,k=30):
    resize = transforms.Resize(256)
    data = np.zeros((len(images),256*256))
    idx = 0
    for img in images:
        data[idx] = np.array(resize(Image.open(io.BytesIO(img))).convert('L')).reshape(1,-1)
        idx = idx+1
    pca = PCA(2)
    converted_data = pca.fit_transform(data)
    pts,indices = graipher(converted_data,k)
    print(indices)
    reduced_images = list(np.array(images)[indices])
    save_reduced_images(reduced_images)
    print(len(reduced_images))
    return reduced_images

def save_reduced_images(images):
    os.makedirs('/content/Pnina/MyDrive/Cycle_GAN/kaggle_dataset/monet_reduced', exist_ok=True)
    i = 0
    for img in images:
        Image.open(io.BytesIO(img)).convert('RGB').save('/content/Pnina/MyDrive/Cycle_GAN/kaggle_dataset/monet_reduced/'+str(i).zfill(3)+'.png')
        i=i+1
