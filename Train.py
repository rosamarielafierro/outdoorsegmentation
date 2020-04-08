import torch
import torch.utils as utils
import torchvision
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


def main(opt, unet_version, ds_version, data_augmentation, inverted_freq, weather):

    params = {'num_epochs': 100,
              'batch_size': 10,
              'num_classes': 19,
              'start_features':32,
              'log_interval':10,
              'iou_threshold': 0.3,
              'adam_learning_rate': 1E-3,
              'adam_aux_learning_rate': 5E-4,
              'adam_weight_decay': 1E-4,
              'sgd_learning_rate': 1E-3,
              'sgd_weight_decay': 1E-4,
              'sgd_momentum': 0.9,
              'device': torch.device("cuda"),
              'dataset_url': '/home/jupyter/it6/utils/',
              'log_dir': '/home/jupyter/it6/runs/',
              'file_suffix':'_split_urls'
              #'auth_service_json_url':'/home/jupyter/it6-oss-9b93ef313e32.json'
              }

    params['device'] = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    da = 'da.' if data_augmentation == 'y' else ''
    inv = 'inv.' if inverted_freq == 'y' else ''
    we = 'we' if weather == 'y'  else ''
    experiment_id =  opt + '.' + unet_version + '.' + ds_version + da  + inv + we

    # Creamos las transformaciones para las imagenes y targets
    """
    tensor_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256, 512), interpolation=0),
        torchvision.transforms.ToTensor()
    ])
    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256, 512), interpolation=0)
    ])
    """
    #Creamos las listas para las transformaciones
    joint_transformations, joint_transformations_vt, img_transformations, img_transformations_vt = [], [], [], []
    
    #Añadimos el Resize
    joint_transformations.append(aux.Resize(256,512))
    joint_transformations_vt.append(aux.Resize(256,512))

    #En caso de Data Augmentation, se añade un Random Vertical Flip y el ajuste de parametros de imagen
    if data_augmentation == "y":
        print('set Data Augmentation\n')
        joint_transformations.append(aux.RandomVerticallyFlip())
        img_transformations.append(torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    
    if weather == 'y':
        print('set weather\n')
        img_transformations.append(aux.Weather())
    
    #añadimos la transformacion final para tensor en las img
    img_transformations.append(torchvision.transforms.ToTensor())
    img_transformations_vt.append(torchvision.transforms.ToTensor())
    
    #Aplicamos la transformacion conjunta sobre img y target
    joint_transforms = aux.JointCompose(joint_transformations)
    joint_transforms_vt = aux.JointCompose(joint_transformations_vt)

    #Aplicamos solo la transformacion sobre img
    img_transforms = torchvision.transforms.Compose(img_transformations)
    img_transforms_vt = torchvision.transforms.Compose(img_transformations_vt)

    
    
    """
    transformations, transformations_target = [], []
    transformations.append(torchvision.transforms.Resize(size=(256, 512),interpolation=0))
    transformations_target.append(torchvision.transforms.Resize(size=(256, 512),interpolation=0))

    if data_augmentation == "y":
        transformations.append(torchvision.transforms.RandomVerticalFlip(p=0.5))
        transformations.append(torchvision.transforms.ColorJitter(brightness=3))
        transformations_target.append(torchvision.transforms.RandomVerticalFlip(p=0.5))

    #añadimos la transformacion final para tensor
    transformations.append(torchvision.transforms.ToTensor())

    #Aplicamos la transformacion sobre target
    tensor_transform = torchvision.transforms.Compose(transformations)
    target_transform = torchvision.transforms.Compose(transformations_target)
    """
    #Definimos temporizadores
    #Arrancamos el temporizador
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_total = torch.cuda.Event(enable_timing=True)
    end_total = torch.cuda.Event(enable_timing=True)

    
    # Creamos los datasets y dataloaders
    train_dataset = MyDataset(version=ds_version, split='train', joint_transform=joint_transforms, img_transform=img_transforms, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'])
    train_loader = utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)

    val_dataset = MyDataset(version=ds_version, split='val', joint_transform=joint_transforms_vt, img_transform=img_transforms_vt, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'])
    val_loader = utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)

    test_dataset = MyDataset(version=ds_version, split='test', joint_transform=joint_transforms_vt, img_transform=img_transforms_vt, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'])
    test_loader = utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    
    
    """
    train_dataset = MyDataset(version=ds_version, split='train', transform=tensor_transform, target_transform=target_transform, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'])
    train_loader = utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers = 4)

    val_dataset = MyDataset(version=ds_version, split='val', transform=tensor_transform, target_transform=target_transform, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'])
    val_loader = utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers = 4)

    test_dataset = MyDataset(version=ds_version, split='test', transform=tensor_transform, target_transform=target_transform, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'])
    test_loader = utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers = 4)
    """

    def train_one_epoch(train_loader, net, optimizer, criterion, hparams):

        # Activate the train=True flag inside the model
        net.train()

        device = hparams['device']
        batch_size = hparams['batch_size']
        train_loss, train_accs = 0, 0
        train_iou = {}
        times_per_step_iteration = []
        times_per_metric_iteration = []
        times_per_iteration = []
        for batch_index, (img, target) in enumerate(train_loader):
        #Arrancamos temporizador general
            start_total.record()
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()

            # Arrancamos temporizador para inferencia
            start.record()
            output = net(img)

            target = target.long()


            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = aux.get_predicted_image(output)

            #Paramos temporizador de inferencia
            end.record()
            torch.cuda.synchronize()
            times_per_step_iteration.append(start.elapsed_time(end))

            # Accuracy
            #Arrancamos temporizador para métricas
            start.record()

            # Desvinculamos el valor de los nuevos targets y los pasamos a CPU para calcular las métricas
            output, target, pred = output.detach().cpu(), target.detach().cpu(), pred.detach().cpu()
            train_loss += loss.item()
            # Devuelve values, indices. Los indices son el nº de feature map (clase) en la que se encuentra el valor más alto en el pixel
            train_accuracy = metrics.calculate_accuracy(output, target) #, predicted
            train_accs  += train_accuracy


            iou_inds = metrics.calculate_iou(pred, target)
            for key in iou_inds:
                if key not in train_iou:
                    train_iou[key] = iou_inds[key]
                else:
                    train_iou[key] += iou_inds[key]

            #Paramos temporizador para métricas
            end.record()
            torch.cuda.synchronize()
            times_per_metric_iteration.append(start.elapsed_time(end))

            #Paramos temporizador general
            end_total.record()
            torch.cuda.synchronize()
            times_per_iteration.append(start_total.elapsed_time(end))

            avg_time_taken = sum(times_per_iteration)/len(times_per_iteration)
            avg_time_step_taken = sum(times_per_step_iteration)/len(times_per_step_iteration)
            avg_time_metrics_taken = sum(times_per_metric_iteration)/len(times_per_metric_iteration)


        print('Average Time spent total: {:.02f}s'.format(avg_time_taken*1e-3))
        print('Average Time spent by steps: {:.02f}s'.format(avg_time_step_taken*1e-3))
        print('Average Time spent by metrics: {:.02f}s'.format(avg_time_metrics_taken*1e-3))
        print('Average Time spent by data load: {:.02f}s'.format(avg_time_taken*1e-3-avg_time_step_taken*1e-3-avg_time_metrics_taken*1e-3))


        train_loss = train_loss / (len(train_loader.dataset) / batch_size)
        train_accs = 100 * (train_accs / (len(train_loader.dataset) / batch_size))
        train_iou = metrics.convert_batched_iou(train_iou, (len(train_loader.dataset) / batch_size))
        mIoU = metrics.get_mIoU(train_iou)
        mIoU_desc = metrics.miou_to_string(train_iou)
        return train_loss, train_accs, mIoU, mIoU_desc

        
    def val_one_epoch(val_loader, net):

        net.eval()
        device = params['device']
        batch_size = params['batch_size']
        val_loss = 0
        val_acc = 0
        val_iou = {}
        pred = 0
        with torch.no_grad():
            for batch_index, (img, target) in enumerate(val_loader):
                img, target = img.to(device), target.to(device)
                output = net(img)
                target = target.long()

                loss = criterion(output, target).item()
                val_loss += loss

                pred = aux.get_predicted_image(output)
                # Desvinculamos el valor de los nuevos targets y los pasamos a CPU para calcular las métricas
                output, target, pred = output.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

                # compute number of correct predictions in the batch
                val_accuracy = metrics.calculate_accuracy(output, target)
                val_acc += val_accuracy
                iou_inds = metrics.calculate_iou(pred, target)

                for key in iou_inds:
                    if key not in val_iou:
                        val_iou[key] = iou_inds[key]
                    else:
                        val_iou[key] += iou_inds[key]
                    #print('Batch index: {}, loss: {}, accuracy: {:.2f}%'.format(batch_index, loss, val_accuracy * 100))
        # Average acc across all correct predictions batches now
        val_loss = val_loss / (len(val_loader.dataset) / batch_size)
        val_acc = 100 * (val_acc / (len(val_loader.dataset) / batch_size))
        val_iou = metrics.convert_batched_iou(val_iou, (len(val_loader.dataset) / batch_size))
        mIoU = metrics.get_mIoU(val_iou)

        #print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.0f}%, mIoU: {:.4f}\n'.format(val_loss,  val_acc, mIoU))
        mIoU_desc = metrics.miou_to_string(val_iou)
        return val_loss, val_acc, mIoU, mIoU_desc



    ## Build the net here
    if unet_version == 'linear':
        print('set linear unet\n')
        unet = UNet(num_classes=params['num_classes'], start_features=params['start_features'])
    else:
        print('set ' + str(unet_version) + ' unet\n')
        unet = UNetFull(num_classes=params['num_classes'], start_features=params['start_features'], bilinear=unet_version == 'bilinear')

    ###################

    writer_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = params['log_dir'] + '/' + experiment_id
    run_data_folder = run_path + '/' + writer_date
    tb_writer_train = SummaryWriter(log_dir= run_data_folder + "/train")
    tb_writer_val = SummaryWriter(log_dir= run_data_folder + "/val")
    images_train, targets_train = next(iter(train_loader))
    images_val, targets_val = next(iter(val_loader))

    aux.write_tensorboard_inicio(tb_writer_train,tb_writer_val, unet, images_train, images_val, targets_val)

    ##################

    unet.to(params['device'])
    net_params = unet.parameters()
    
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

    best_epoch_miou, added_targg = 0, False

    print('Dataset train images: {}, dataset val images: {}'.format(len(train_loader.dataset), len(val_loader.dataset)))

    train_losses, train_acc_hist, val_losses, val_acc_hist, mIoU_hist_train, mIoU_hist_val = [], [], [], [], [], []
    for epoch in range(1, params['num_epochs'] +1):

        # Compute & save the average training loss for the current epoch
        print('#################### Epoch: {} ####################\n'.format(epoch))

        aux.print_timestamp('Inicio training epoch {}'.format(epoch))
        train_loss, train_acc, train_mIoU, mIoU_desc_train = train_one_epoch(train_loader, unet, optimizer, criterion, params)
        print('Training set: Average loss {:.4f}, Average accuracy {:.2f}%, mIoU: {:.2f}\n{}\n'.format(train_loss, train_acc, train_mIoU, mIoU_desc_train))

        aux.print_timestamp('Inicio validacion epoch {}'.format(epoch))
        val_loss, val_acc, val_mIoU, mIoU_desc_val = val_one_epoch(val_loader, unet)
        print('Validation set: Average loss: {:.4f}, Mean accuracy: {:.2f}%, mIoU: {:.2f}\n{}\n'.format(val_loss, val_acc, val_mIoU, mIoU_desc_val))

        train_mAP = sum(train_acc_hist) / epoch # params['num_epochs']
        val_mAP = sum(val_acc_hist) / epoch # params['num_epochs']

        if val_mIoU > best_epoch_miou:
            best_epoch_miou = val_mIoU
            print('Guardamos el modelo en epoch {} ( mIoU {:.2f})'.format(epoch, val_mIoU))
            aux.save_model(unet, run_data_folder +'/best')
            aux.write_tensorboard_best_IoU(tb_writer_val, val_mIoU, epoch)
        #train_losses.append(train_loss)
        train_acc_hist.append(train_acc)
        #val_losses.append(val_loss)
        val_acc_hist.append(val_acc)
        #mIoU_hist.append(val_mIoU)

        mIoU_hist_train.append(train_mIoU)
        mIoU_hist_val.append(val_mIoU)

        images_val = images_val.to(params['device'])
        predicted_output = unet(images_val)

        aux.write_tensorboard_epoch(tb_writer_train, tb_writer_val,run_data_folder, predicted_output[0], epoch, train_loss, train_acc, val_loss, val_acc, train_mIoU, val_mIoU, train_mAP, val_mAP)
    aux.save_model(unet, run_data_folder +'/last')
    print('Fin del entrenamiento\n')

    tb_writer_train.close()
    tb_writer_val.close()

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


    unet = aux.load_model(run_data_folder +'/best')
    print('Dataset test images: {}'.format(len(test_loader.dataset)))
    test_loss, test_acc, mIoU, mIoU_desc = test_model(test_loader, unet)
    print('Test set: Average loss: {:.4f}, Mean accuracy: {:.2f}%, mIoU: {:.2f}%\n{}\n'.format(test_loss, test_acc, mIoU, mIoU_desc))


"""
Los argumentos son son:
    ~ Optimizers (-opt)
        - adam
        - sgd
    ~ UNet (-net)
        - full
        - bilinear
        - trans
    ~ Dataset (-ds)
        - cityscapes
        - it6
    ~ Data Augmentation (-da)
        - y
        - n
    ~ Inverted Frequency (-invf)
        - y
        - n
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--optimizer', default='adam', help='Optimizer for the execution')
    parser.add_argument('-net', '--network', default='trans', help='UNet version (with/out concatenations)')
    parser.add_argument('-ds', '--dataset', default='cityscapes', help='cityscapes full dataset, it6 skim version')
    parser.add_argument('-da', '--data_augmentation', default='n', help='Apply data augmentation, random vertical flip and color jitter')
    parser.add_argument('-invf', '--inverted_freq', default='n', help='Apply inverted frequency to the criterion')
    parser.add_argument('-weather', '--weather', default='n', help='Apply weatherdata augmentation, cloud, rain, snow and fog')
    args = parser.parse_args()
    print('{}, {}, {}, {}, {}, {}'.format(args.optimizer, args.network, args.dataset, args.data_augmentation, args.inverted_freq, args.weather))

    main(opt= args.optimizer, unet_version=args.network, ds_version=args.dataset, data_augmentation=args.data_augmentation, inverted_freq=args.inverted_freq, weather=args.weather)
    