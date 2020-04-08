import torch
import numpy as np


# Metrics

def calculate_accuracy(input, target, ignore_index=255):

  _, predicted = torch.max(input.data, 1)
  #print(predicted.shape)
  total_pixels_train = target.nelement()
  pixels_to_ignore = predicted.eq(ignore_index).sum().item()
  #print('Pixels que son iguales: {}'.format(predicted.eq(target.data).sum().item()))
  #print('Pixels a ignorar (255): {}'.format(pixels_to_ignore))
  # Aqui estamos restando los pixeles que son 255, de modo que el accuracy va a salir un poco más baja en caso de que haya algún 255.
  # Tenemos que coger el segundo valor en caso de que el primero sea un 255?
  correctos_batch, fallidos_batch = predicted.eq(target.data).sum().item() - pixels_to_ignore, predicted.ne(target.data).sum().item()
  train_accuracy = correctos_batch / total_pixels_train
  return train_accuracy #, predicted



def calculate_iou(prediction, target):

  num_classes = np.unique(prediction)
  num_classes_target = np.unique(target)
  class_iou = {}
  #for class_id in num_classes_target:
  for class_id in range(19):
    #if class_id != 255:
      # Tenemos un array booleano por cada clase indicando si ese pixel corresponde a la clase. Para la predicción y para el target
      pred_inds = prediction == class_id
      target_inds = target == class_id
      intersection = np.logical_and(pred_inds, target_inds).long().sum().item()
      union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection 
      if(union == 0):
        iou = 0
      else:
        iou = intersection/union
      class_iou[class_id] = iou
  
  return class_iou


def convert_batched_iou(input_iou, length):

  for key in input_iou:
      input_iou[key] = input_iou[key] / length

  return input_iou

def get_mIoU(input_iou):

  count = 0
  _sum = 0
  for key in input_iou:
      count += 1
      _sum += input_iou[key]

  return _sum/count


def miou_to_string(iou_classes):

  sorted_iou = {k: v for k, v in sorted(iou_classes.items(), key=lambda item: item[1], reverse=True)}
  string = ['IoU per class\n']
  #string.append('---------------------\n')
  string.append("Classes\t Values\n")
  string.append('-------   ------\n')
  bprinted_threshold= False
  for key, value in iou_classes.items():    
    """
    if value < iou_threshold and bprinted_threshold == False:
      bprinted_threshold = True
      string.append('--- Threshold ({}) ---\n'.format(iou_threshold))
    """
    string.append(str(key) + '\t' + str(' %.2f' % value) + '\n')
  
  return ''.join(string)

"""
def write_tensorboard(tb, epoch, train_loss, train_acc, val_loss, val_acc, mIoU_val, improvement_pred):
    tb.add_scalar('Loss/Train', train_loss, epoch)
    tb.add_scalar('Loss/Val',  val_loss, epoch)
    tb.add_scalar('Accuracy/Train', train_acc, epoch)
    tb.add_scalar('Accuracy/Val', val_acc, epoch)
    tb.add_scalar('mIoU (val)', mIoU_val, epoch)
    improvement_pred = torch.Tensor(improvement_pred)
    improvement_pred = improvement_pred.view(3, 256, 512)
  
    #grid = torchvision.utils.make_grid([gt,pred,improvement_pred])
    #tb.add_image('Predictions evolution',grid)
    tb.add_image('Predictions evolution', improvement_pred, epoch)
"""
