import torch
import torch.utils as utils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import random

import imgaug as ia
import imgaug.augmenters as iaa

# SERIALIZATION

def save_model(model, path):
    torch.save(model, path)# model.state_dict()

def load_model(path):
    return torch.load(path)

# AUXILIAR FUNCTIONS

def show_image(pil_image):
    plt.imshow(np.asarray(pil_image))

def get_predicted_image(input):
    predicted = torch.argmax(input.data, dim = 1)
    return predicted

def get_predicted_single_image(input):
  
    predicted = torch.argmax(input.data, dim=0)
    return predicted

def print_timestamp(input):
    print(input + " " + datetime.utcnow().strftime("%H:%M:%S"))

def target_to_rgb(target, num_classes = 19):

      #Create color map
    colormap = np.zeros((19, 3), dtype = np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    # Create empty rgb channels
    r = np.zeros_like(target).astype(np.uint8)
    g = np.zeros_like(target).astype(np.uint8)
    b = np.zeros_like(target).astype(np.uint8)

    # Find / Set correct rgb values
    for l in range(0, num_classes):
        idx = target == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]

    # Concatenate channels
    rgb = np.stack([r, g, b], axis = 0)

    return rgb

def print_graphs(train_losses, val_losses, train_acc_hist, val_acc_hist):
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.legend()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')

    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.plot(train_acc_hist, label='train')
    plt.plot(val_acc_hist, label='val')

def write_tensorboard_inicio(tb_writer_train, tb_writer_val, model, images_train, images_val, targets_val):

    #TODO Only valid when using Unet
    #tb_writer_train.add_graph(model, images_train)

    image = images_val[0]
    target = targets_val[0]
    image = image.cpu()

    target = target.cpu()
    target = target_to_rgb(target)
    
    tb_writer_val.add_image('Ground Truth', image)
    tb_writer_val.add_image('Ground Truth Labelled', target)

    tb_writer_train.flush()
    tb_writer_val.flush()
    
    #tb_writer_train.add_graph(unet, images_train)
    

def write_tensorboard_epoch(tb_writer_train, tb_writer_val, pred_path, predicted, epoch, train_loss, train_acc, val_loss, val_acc, train_mIoU, val_mIoU, train_mAP, val_mAP):

    tb_writer_train.add_scalar('Loss', train_loss, epoch)
    tb_writer_val.add_scalar('Loss', val_loss, epoch)
    tb_writer_train.add_scalar('Accuracy', train_acc, epoch)
    tb_writer_val.add_scalar('Accuracy', val_acc, epoch)
    tb_writer_train.add_scalar('mIoU', train_mIoU, epoch)
    tb_writer_val.add_scalar('mIoU', val_mIoU, epoch)
    tb_writer_train.add_scalar('mAP', train_mAP, epoch)
    tb_writer_val.add_scalar('mAP', val_mAP, epoch)

    predicted_image = get_predicted_single_image(predicted)

    predicted_image = predicted_image.cpu()
    predicted_image_rgb = target_to_rgb(predicted_image)

    tb_writer_val.add_image('Predictions evolution', predicted_image_rgb, epoch)
    
    #predicted_image_rgb = predicted_image_rgb.reshape(256, 512, 3)
    #pred_image_pil = Image.fromarray(predicted_image_rgb)
    #pred_image_pil.save(pred_path + "/predictions/" + str(epoch) + ".png")

    tb_writer_train.flush()
    tb_writer_val.flush()

def write_tensorboard_best_IoU(tb_writer, mIoU, epoch):

    #sacar el maximo de cada mIoU
    mIoU = round(mIoU, 2)
    tb_writer.add_text("Best mIoU", str(mIoU), epoch)
    tb_writer.flush()
    
    
    
class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        assert img.size == target.size
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        return img, target
    
class RandomVerticallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img, target):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM), target.transpose(Image.FLIP_TOP_BOTTOM)
        return img, target

class Resize(object):
    def __init__(self, h=256, w=512):
        self.h = h
        self.w = w
    def __call__(self, img, target):
        transform = torchvision.transforms.Resize(size=(self.h, self.w),interpolation=0)
        return transform(img), transform(target)
    
class Weather(object):
    def __init__(self):
        #self.h = h
        a=True
    def __call__(self, img):
        
        seq = iaa.SomeOf((1, 2),[
            iaa.weather.Snowflakes(),
            iaa.weather.Rain(),
            iaa.weather.Fog(),
            iaa.weather.Clouds()
        ], random_order=True)
        # Order Axis
        
        img = img.convert("RGB")        
        
        img = np.array(img)
        aug = seq(image=img)
        #aug = Image.fromarray(aug)
        #aug.save("testimg.png")
        return aug