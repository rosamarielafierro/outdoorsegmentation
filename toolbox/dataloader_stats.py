import torch
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image
import numpy as np
from collections import Counter
from skimage import io
import os
from glob import glob
import sklearn
from sklearn import metrics
from collections import defaultdict
from __future__ import print_function, absolute_import, division
from collections import namedtuple

torch.manual_seed(1)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(1)

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


#Create dictionaries for the counts
train_classes_counts = defaultdict(int)
val_classes_counts = defaultdict(int)

##TRAIN
#Using torch.unique to count the number of pixels of each class
for i in range(len(train_dataset)):
#for i in range(3):
  #Get the labeled image
  _,train_sample = train_dataset[i]
  train_sample.cpu()
  train_labels,train_counts = torch.unique(train_sample,return_counts=True)
  #Transform to array
  train_labels = train_labels.numpy()
  train_counts = train_counts.numpy()

  #Añadimos el numero de ocurrencias por cada clase
  for j in range (len(train_counts)):
    train_classes_counts[train_labels[j]] += train_counts[j]

print("Number of images in training dataset",len(train_dataset))
print(train_classes_counts)

##VALIDATION
#Using torch.unique to count the number of pixels of each class
for i in range(len(val_dataset)):
  #Get the labeled image
  _,val_sample = val_dataset[i]
  val_sample.cpu()
  val_labels,val_counts = torch.unique(val_sample,return_counts=True)
  #Transform to array
  val_labels = val_labels.numpy()
  val_counts = val_counts.numpy()

  #Añadimos el numero de ocurrencias por cada clase
  for j in range (len(val_counts)):
    val_classes_counts[val_labels[j]] += val_counts[j]

print("Number of images in validation dataset",len(val_dataset))
print(val_classes_counts)

labels = [trainId2label[0].name,
          trainId2label[1].name,
          trainId2label[2].name,
          trainId2label[3].name,
          trainId2label[4].name,
          trainId2label[5].name,
          trainId2label[6].name,
          trainId2label[7].name,
          trainId2label[8].name,
          trainId2label[9].name,
          trainId2label[10].name,
          trainId2label[11].name,
          trainId2label[12].name,
          trainId2label[13].name,
          trainId2label[14].name,
          trainId2label[15].name,
          trainId2label[16].name,
          trainId2label[17].name,
          trainId2label[18].name,
          trainId2label[255].name
          ]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

bar_train = []
bar_val = []
total_train = 0
total_val = 0

#We should order the values for plotting
for i in range(20):
  if i==19:
    i=255
  bar_train.append(train_classes_counts[i])
  total_train += train_classes_counts[i]

for i in range(20):
  if i==19:
    i=255
  bar_val.append(val_classes_counts[i])
  total_val += val_classes_counts[i]

print("Numero de ocurrencias train:")
print(bar_train)


#Normalize the values for comparable results in the graph
bar_train = [bar_train[i]/total_train for i in range(len(bar_train))]
bar_val = [bar_val[i]/total_val for i in range(len(bar_val))]

#Plot the results
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, bar_train, width, label='Train', align='center')
rects2 = ax.bar(x + width/2,  bar_val, width, label='Validation', align='center')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Frequency')
ax.set_title('Classes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects:
        height = round(rect.get_height(),3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)


fig.tight_layout()
fig.set_size_inches(20,10)

plt.show()
