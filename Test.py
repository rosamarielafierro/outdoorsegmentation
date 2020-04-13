import torch
import torch.utils as utils
import torchvision
from torchvision import transforms
from torch import nn
from torch import backends
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image
import numpy as np
import argparse
import os
import math
import time as timer
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from skimage import io

import imgaug as ia
import imgaug.augmenters as iaa

# Reproducibility: asignar la misma seed a todas las librerías para que a ejecuciones iguales obtengamos
# resultados iguales
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
np.random.seed(1)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

from Dataset import MyDataset
import utils as aux
import metrics as metrics
from UNet import UNet
from UNetModes import UNetFull


def main(opt, model_version, ds_version, path, name, inverted_freq, weather):

    params = {'num_epochs': 60,
              'batch_size': 10,
              'num_classes': 19,
              'start_features':32,
              #'log_interval':10,
              #'iou_threshold': 0.3,
              'adam_learning_rate': 1E-3,
              'adam_aux_learning_rate': 5E-4,
              'adam_weight_decay': 1E-4,
              'sgd_learning_rate': 0.1,
              'sgd_weight_decay': 1E-4,
              'sgd_momentum': 0.9,
              'device': torch.device("cuda"),
              'dataset_url': '/home/jupyter/it6/utils/',
              'log_dir': '/home/jupyter/it6/runs/',
              'file_suffix':'_split_urls'
              }
    
    def test_model(test_loader, net):

        net.eval()
        device = params['device']
        batch_size = params['batch_size']
        test_loss = 0
        test_acc = 0
        test_iou = {}
        with torch.no_grad():
            for batch_index, (img, target) in enumerate(test_loader):
                img, target = img.to(device), target.to(device)
                
                if model_version == 'deeplab':
                    output = net(img)['out']
                else:
                    output = net(img)
                
                target = target.long()
                loss = criterion(output, target).item()
                test_loss += loss
                
                pred = aux.get_predicted_image(output)
                
                output, target, pred = output.detach().cpu(), target.detach().cpu(), pred.detach().cpu()                
                # compute number of correct predictions in the batch
                test_accuracy = metrics.calculate_accuracy(output, target)
                test_acc += test_accuracy
                
                iou_inds = metrics.calculate_iou(pred, target)

                for key in iou_inds:
                    if key not in test_iou:
                        test_iou[key] = iou_inds[key]
                    else:
                        test_iou[key] += iou_inds[key]

        test_loss = test_loss / (len(test_loader.dataset) / batch_size)
        test_acc = 100 * (test_acc / (len(test_loader.dataset) / batch_size))
        test_iou = metrics.convert_batched_iou(test_iou, (len(test_loader.dataset) / batch_size))
        mIoU = metrics.get_mIoU(test_iou)

        mIoU_desc = metrics.miou_to_string(test_iou)
        return test_loss, test_acc, mIoU, mIoU_desc
    
    
    
    #Creamos las listas para las transformaciones
    joint_transformations_vt, img_transformations_vt = [], []
    
    #Añadimos el Resize
    joint_transformations_vt.append(aux.Resize(256,512))
    
    #añadimos la transformacion final para tensor en las img y normalizamos
    
    img_transformations_vt.append(torchvision.transforms.ToTensor())
    
    #In the case of DeepLabv3, we apply the recommended normalization
    if model_version == 'deeplab':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_transformations_vt.append(torchvision.transforms.Normalize(mean, std))
    
    #Aplicamos la transformacion conjunta sobre img y target
    joint_transforms_vt = aux.JointCompose(joint_transformations_vt)

    #Aplicamos solo la transformacion sobre img
    img_transforms_vt = torchvision.transforms.Compose(img_transformations_vt)
    
    test_dataset = MyDataset(version=ds_version, split='test', joint_transform=joint_transforms_vt, img_transform=img_transforms_vt, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'], add_weather= weather == 'y')
    test_loader = utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    
    model = aux.load_model(path +'/'+name)
    model.to(params['device'])
    net_params = model.parameters()
    
    #Depending on the inverted frequency parameter we apply this parameter as a weight to balance the Loss Function
    if inverted_freq == 'y':
        print('set Inverted Frequency weights \n')
        num_pixels_per_class = [127414939, 21058643, 79041999, 2269832, 3038496, 4244760, 720425, 1911074, 55121339, 4008424, 13948699, 4204816, 465832, 24210293, 925225, 813190, 805591, 341018, 1430722]
        inverted_weights = [(1/num_pixels) for num_pixels in num_pixels_per_class]
        inverted_weights = torch.FloatTensor(inverted_weights).to(params['device'])
        criterion = torch.nn.CrossEntropyLoss(weight=inverted_weights, ignore_index=255)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)


    if opt == 'adam':
        print('set adam optimizer\n')
        optimizer = torch.optim.Adam(net_params, lr=params['adam_learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=params['adam_weight_decay'], amsgrad=False)
    elif opt == 'sgd':
        print('set SGD optimizer\n')
        optimizer = torch.optim.SGD(net_params, lr=params['sgd_learning_rate'],momentum= params['sgd_momentum'], weight_decay=params['sgd_weight_decay'])
    
    
    print('Dataset test images: {}'.format(len(test_loader.dataset)))
    test_loss, test_acc, mIoU, mIoU_desc = test_model(test_loader, model)
    print('Test set: Average loss: {:.4f}, Mean accuracy: {:.2f}%, mIoU: {:.2f}%\n{}\n'.format(test_loss, test_acc, mIoU, mIoU_desc))


"""
Los argumentos son son:
    ~ Path (-path)
    ~ Name (-n)
    

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--optimizer', default='adam', help='Optimizer for the execution')
    parser.add_argument('-path', '--path', default='', help='Path for the model')
    parser.add_argument('-n', '--name', default='best', help='Name of the model file')
    parser.add_argument('-net', '--network', default='trans', help='Model version (with/out concatenations), Unet or DeepLabv3')
    parser.add_argument('-ds', '--dataset', default='cityscapes', help='cityscapes full dataset, it6 skim version')
    parser.add_argument('-invf', '--inverted_freq', default='n', help='Apply inverted frequency to the criterion')
    parser.add_argument('-weather', '--weather', default='n', help='Apply weatherdata augmentation, cloud, rain, snow and fog')

    args = parser.parse_args()
    print('{}, {}, {}, {}, {}, {}, {}'.format(args.optimizer, args.network, args.dataset, args.path, args.name, args.inverted_freq, args.weather))

    main(opt= args.optimizer, model_version=args.network, ds_version=args.dataset, path= args.path, name=args.name, inverted_freq=args.inverted_freq, weather=args.weather)

    
def test_model(test_loader, net):

        net.eval()
        device = params['device']
        batch_size = params['batch_size']
        test_loss = 0
        test_acc = 0
        test_iou = {}
        with torch.no_grad():
            for batch_index, (img, target) in enumerate(test_loader):
                img, target = img.to(device), target.to(device)
                
                if model_version == 'deeplab':
                    output = net(img)['out']
                else:
                    output = net(img)
                
                target = target.long()
                loss = criterion(output, target).item()
                test_loss += loss
                
                pred = aux.get_predicted_image(output)
                
                output, target, pred = output.detach().cpu(), target.detach().cpu(), pred.detach().cpu()                
                # compute number of correct predictions in the batch
                test_accuracy = metrics.calculate_accuracy(output, target)
                test_acc += test_accuracy
                
                iou_inds = metrics.calculate_iou(pred, target)

                for key in iou_inds:
                    if key not in test_iou:
                        test_iou[key] = iou_inds[key]
                    else:
                        test_iou[key] += iou_inds[key]

        test_loss = test_loss / (len(test_loader.dataset) / batch_size)
        test_acc = 100 * (test_acc / (len(test_loader.dataset) / batch_size))
        test_iou = metrics.convert_batched_iou(test_iou, (len(test_loader.dataset) / batch_size))
        mIoU = metrics.get_mIoU(test_iou)

        mIoU_desc = metrics.miou_to_string(test_iou)
        return test_loss, test_acc, mIoU, mIoU_desc