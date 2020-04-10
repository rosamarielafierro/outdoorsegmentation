## Repository Structure

The repository is structured as follows:
 - Cityscapes - folder containing cityscapes dataset.
 - Figures - folder containing project images.
 - Dataset.py - python file containing the dataset class.
 - Train.py - python file containing the train class.
 - Unet.py - python file containing the Unet class.
 - UnetModes.py - python file containing extra Unet modes.
 - metrics.py - python file containing metric calculators.

# Outdoor Semantic Segmentation

### About
 - Date: 14/4/2020
 - Authors: Alberto Masa, Fernando Tibet, Mariela, Sina Sadeghi
 - Institute: Universitat Politecnica De Cataluña

## Motivation

## Proposal

## Milestones

## Cityscapes Dataset
The cityscapes dataset includes a diverse set of street scene image captures from 50 different cities. The dataset includes semantic, instance-wise, and dense pixel annotation for a total of 19 classes. The dataset includes 5,000 images at a resolution of 1024x2048. 

![Data Sample](https://www.researchgate.net/profile/Varun_Jampani/publication/319056828/figure/fig3/AS:667765274845187@1536219056188/Qualitative-results-from-the-Cityscapes-dataset-Observe-how-NetWarp-PSPNet-is-able-to.jpg)


### Dataset Class
We created a custom Dataset class capable of loading the images and targets from the Cityscapes dataset. However, in order to increase flexibility, a helper function "Split_Generator" was written to produce a .csv file containing the URLS with the images and targets. The dataset class uses the .csv to load the data and is, therefore, not limited to only Cityscapes.

(insert dataset class code screenshot)

### Data Loader
A data loader helper function was written in order to facilitate data retrieval and to pre-process the data in preparation for training. 

![Data Sample](https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/stuttgart01-2040x500.png)

#### Metrics
A set of functions were created in the dataset class in order to calculate the statistics of the train and validation splits in order to track progress throughout the development of the network. The accuracy, ~~precision, mean average precision~~, intersection over union, and mean intersection over union were gathered, always ignoring the label 255 of the Cityscapes dataset. 


## Architectures
### Network
The U-net is a fully convolutional network created specifically for computer vision problems, including segmentation tasks. It became popular because of its efficient use of the GPU and its ability to learn with a limited dataset. What makes the U-net noble from other network architectures is that every pooling layer is mirrored by an up-sampling layer. The following figure shows the U-net contracting path (left side) and an expansive path (right side), both of which are symmetrically structured.

![Unet Model Diagram](https://github.com/it6aidl/outdoorsegmentation/blob/master/Unet%20-%20Basic.png)

This allows the network to reduce spatial information while increasing feature information on its way down, and reduce feature information while increasing station information on the way up, leading to highly efficient image processing. The U-net class is defined as follows,

(Insert Unet class code screenshot)

### Optimizer
(help)

### Concatenation Layer
Since the U-net downsamples the feature information in the first half of the network, there is a risk of loosing valuable information. To overcome this, we concatenated all the feature maps in the decoding layers with the feature maps from the encoding layers. This assures that any information learned in the ignitions layers will be retained throughout the network. 

![Concat Layers](https://github.com/it6aidl/outdoorsegmentation/blob/master/Unet%20-%20Concat.png)

### Bi-linear Interpolation
As the input images run through the encoding layers they loose their position information, which is vital for segmentation tasks. Therefore, after 5 layers of pooling the network has 5 layers of up-sampling to return the image to its original dimensions. For the initial attempt, bi-linear interpolation was used as the upsampling technique. 

Bi-linear interpolation, or bi-linear filtering, interpolating along two axis in order to upsample data. 

![Bi-linear Interpolation](https://www.oreilly.com/library/view/deep-learning-for/9781788295628/assets/a4df8c96-4e64-450f-b891-9efb18fc7368.png)

### Transposed Convolutions
Transposed convolutions, or deconvolution, are another upsample tequnique that makes use of learnable parameters to allow the network to decide how to map the pixels and fill in the dots efficiently during the upsampling. Transposed convolutions can be thought of as inverted convolutions, mapping one neuron to many neurons, as opposed to many neurons to one neuron.

![Transposed Convolutions](http://d2l.ai/_images/trans_conv.svg)

The following code snippet shows how the transposed layers where defined,
(pic)
as well as how the layers were implemented in the network's forward function,
(pic)

## Results
Throughout the process of advancing the network, multiple experiments were conducted in order to track progress. 


### Experiment 1: Base-line
The first experiment was intended to act as a base-line for the all future experiments. It consisted of the network training with the Adam optimizer. 
(statistics image)

### Experiment 2: Stochastic Gradient Descent
>>>>>>> 29faa2950c1d0c26a71f656cdcaf98b384b58370
For the second experiment, the Adam optimizer was replaced with the SGD optimizer. 
(statistics image)

### Experiment 3: Data Augmentation
A data augmentation class was created in order to expand our dataset. This experiment featured the SGD optimizer coupled with flips and random noise.
(statistics image)

### Experiment 4: Inverted Weight
Weights were added to the loss using the inverted frequency. Using the information obtained from train split, the inverted frequency was calculated for each class.
(statistics image)

### Experiment 5: Weather Augmentation
In order for the network to prepare for varying road scenarios, in this experiment, it was trained while running the weather augmentation online to generate rain and snow. 
(statistics image)

### Experiment 6: Deep Lab
Finally, the entire project was run using the pre-w eighted Deep Lab model to compare the results to our U-net.
(statistics image)


### Metrics

Evaluating and comparing the experiments is a nuclear part of the scientific work and opens the path to adjust parameters and propose changes. For this project we defined several metrics to compare models and trainings

#### Accuracy
The main metric we used to evaluate the experiments if the accuracy of the model configuration. The model prediction accuracy  is calculated dividing the number of encerted pixels by the number of total pixels. However, there is a class that we are ignoring throughout the experiments and does not compute for the accuracy.
Next we show the accuracy comparison: 

*(mpl accuracy graph when we have all the experiments)*


#### IoU per class
The previous metric is a generalization of how our model works overall. This next one gives emphasis on the nature of the data. Intersection over Union (Jaccard score) is an effective, well known metric for pixel classification in object segmentation problems. The IoU score for each class is a number between 0 and 1 calculated by dividing the intersection from the pixels in prediction and GT by the union of these two surfaces.

As we presented in the dataset statistics, we have a noticeable class imbalance, which ends up in an unbalanced IoU. The classes that appear the most in the dataset (pavement, sky) reach a higher IoU than the ones that appear very few times (signals, traffic lights)

*(paste text IoU per class in best val epoch whichever exp)*


#### mIoU

The other metric that illuminated our grievous path through the fathomless darkness of semantic segmentation was the mean Intersection over Union. A mean calculation of every class IoU is used to measure how well is in classificating all the classes.

*(mpl mIoU graph when we have all the experiments)*

#### Test results

The previous metrics were taken in the validation phase of our training. Concluding the experiment we test the model configuration with the test dataset. The results in this phase give us an overall understanding of the performance. 
*(excel table of the results)*

| Optimizer (LR) | Model | Version | Configuration | Accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) |  UNet| Linear| | 76.7| 38
|  |  | Bilinear 3x3/1| | 77.9| 40
|  |  | Bilinear 1x1| |78.5 | 42
|  |  | Transpose| | 78.3| 43
|  |  | Transpose|DA |77.3 | 39
|  |  | Transpose|DA & IF |70.16 |34 
|  |  | Transpose|Weather DA | | 
|  | Deeplabv3 | | |76.86 | 39
| SGD (0.001) | UNet | Transpose| |75.3 | 31
|  | Deeplabv3 | | | | 
| SGD (0.1) | UNet| Transpose| | | 
|  | Deeplabv3 | | | | 
|  | Deeplabv3 | |Weather DA | | 

## Conclusion

(we should decide what out conclusions for the project are and insert here)

## Future Work

 - Driveable Zone Segmentation
 - Pruning
 - Focal Loss

## References