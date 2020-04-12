


## Repository Structure

The repository is structured as follows:
 - Figures - folder containing project images.
 - Dataset.py - python file containing the dataset class.
 - Train.py - python file containing the train class.
 - Unet.py - python file containing the Unet class.
 - UnetModes.py - python file containing extra Unet modes.
 - metrics.py - python file containing metric calculators.

# Outdoor Semantic Segmentation

### About
 - Date: 14/4/2020
 - Authors: Alberto Masa, Fernando del Tíbet, Mariela Fierro, Sina Sadeghi
 - Institute: Universitat Politecnica De Cataluña

## Motivation

Our field of interest was driving-related topics. This includes a deeper understanding of image-video processing for understanding the context around the vehicle to improve safety.

## Proposal
 - [x] Analyze the data provided in the selected dataset and adapt it to be used in a Semantic Segmentation network.
 - [x] Mitigate the class imbalance based on a better understanding of our data.
 - [x] Learn how to do a transfer learning from the previous task to another one, for instance, detecting the drivable area.
 - [x] Reproduce a semantic segmentation network described in the U-Net paper from scratch.
 - [x] Apply data augmentation, generate different kinds of weather such as fog, rain, snowflakes.

## Milestones

 - [x] Obtain and process the Cityscapes dataset.
 - [x] Train a semantic segmentation network and analyze the results.
 - [x] Use weighted Loss.
 - [x] Apply data augmentation.
 - [ ] Use model for selecting drivable area.

## Dataset
The Cityscapes dataset includes a diverse set of street scene image captures from 50 different cities around the world designed specifically for training segmentation models. The dataset includes semantic, instance-wise, and dense pixel annotation for a total of 30 classes. The dataset consists of 5,000 images at a resolution of 1024x2048.

![Cityscapes](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Cityscapes.png)

A custom dataset class capable of loading the images and targets from the Cityscapes dataset was created. The following functions where included,

-   init( ) - initializes dataset object.
-   len( ) - returns dataset size.
-   getitem( ) - loads and transforms image/target.

The loaded images are resized to 256x512 and converted to tensors during the transformation. The loaded targets are resized to 256x512, with the interpolation parameter set to 0. 

A snippet of the transformation code is presented below,

![Transformation Code](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Transform%20Code.png)

In order to increase the flexibility of the network, a "Split_Generator" was created to produce a .csv file containing the URLS linking the dataset images and targets. The dataset class uses the .csv to locate the data in preparation for training. The following number of images and targets where used for each split:

-   Test: 250
    
-   Validation: 250
    
-   Train: 1600
    

  
As suggested by the Cityscapes documentation, classes with a label id of 255 were emitted, resulting in a total of 19 distinct classes: Road, Sidewalk, Building, Wall, Fence, Pole, Traffic Light, Traffic Sign, Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, and Bicycle.
    
A set of functions were created in the dataset class in order to calculate the statistics of the train and validation splits. The pixel accuracy,

![Pixel Accuracy Formula](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Accuracy%20formula.png)

intersection over union,

![IoU Formula](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/IoU%20formula.png)

and mean intersection over union, which averages the IoU along every class, were gathered and used to track the progress throughout the development of the network.

## Architectures
### Network
The U-net is a fully convolutional network created specifically for computer vision problems, including segmentation tasks. It became popular because of its efficient use of the GPU and its ability to learn with a limited dataset. What makes the U-net noble from other network architectures is that every pooling layer is mirrored by an up-sampling layer. The following figure shows the U-net contracting path (left side) and an expansive path (right side), both of which are symmetrically structured.

![Unet Model Diagram](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Unet.png)

This allows the network to reduce spatial information while increasing feature information on its way down, and reduce feature information while increasing station information on the way up, leading to highly efficient image processing. 

### Optimizer
An optimizer is necessary for minimizing the loss function. In this project both the Adaptive Moment Estimation (Adam) and Stochastic Gradient Descent (SGD) optimizers were tested. SGD calculates the gradient descent for each example, reducing batch redundancies, and improving computational efficiency. Adam is a mixture of the SGD and RMSprop optimizers, and offers an adaptive learning rate, increasing the network's flexibility.

### Concatenation Layer
Since the U-net downsamples the feature information in the first half of the network, there is a risk of loosing valuable information. To overcome this, we concatenated all the feature maps in the decoding layers with the feature maps from the encoding layers. This assures that any information learned in the ignitions layers will be retained throughout the network. 

![Concat Layers](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Unet%2BConcat.png)

### Bi-linear Interpolation
In order to recover the original input resolution at the output of the network, a bi-linear interpolation was performed. For bi-linear interpolation, a weighted average is calculated on the four nearest pixels to achieve a smooth output. The data was interpolated along 2-axis during upsampling, following the following formula,

![Bilinear Formula](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Bilinear%20Interpolation.png)

where f(x,y) is the unknown function, in our case pixel intensity, and the four nearest points being,

![Bilinear Formula2](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Bilinear%20Interpolation2.png)

### Transposed Convolutions
To improve the quality and efficiency of the upsampling, the bi-linear interpolation was replaced by transposed convolutions. These convolutions make use of learneable parameters to enable the network to “learn” how to best transform each feature map on its own.

## Results
Throughout the process of advancing the network, multiple experiments were conducted in order to track progress. What drove us during the whole process of experimentation was evidence :eyes:. What we searched over and over was improvement :sunrise_over_mountains:. But we didn't find it :trollface:, at least how we expected. We expected to see results as we have been seeing in the labs and in the ML books, but maybe we had a stroke of reality. This is the nature of science and engineering :smirk:. We stuck with the configuration that gave us better results until the moment and build on top of it.
The first 6 experiments consists of adjusting configurations for the network itself and adding ML techniques. For the last two experiments we wanted to play a little bit and test the theoretical part we learned in class: we changed the optimizer for our network and an hyperparameter, the learning rate.

**Fer: pongo algunas gráficas y tablas y después con Albert&co vemos cual son más significativas **

### Experiment 1: Linear UNet
The first experiment was intended to act as a base-line for the all future experiments. It consisted of a linear version (removing the concatenations) of the UNet using Adam optimizer. This lightweight version works for us to see a quick segmentation result that is easy to understand and easy to be improved by adding components to the configuration. On the other hand it gives very little precision to the prediction. The architecture is presented in the figure #1 and the method to upsample is *torch.nn.Upsample*. 


![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamlinearloss.png)


### Experiment 2: Concat UNet
For the second experiments, we improved the network to embrace the concatenations defined in the canonical net. Also, we implemented another way to upsample the encoded data in the net through Transposed convolutions.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adambilinearloss.png)


#### Transpose Kernel 3 Padding 1 vs Transpose Kernel 1
*Explicar porqué hemos hecho esto*


    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding= 1))
  ---

    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1))
![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransloss.png)


### Experiment 3: Change optimizer

After running different configurations, we chose the one to perform the rest of the experiments: transposed convolutions UNet. This gave us the better validation results (by a small margin though) than the upsampling version.
Then, before introducing the ML techniques, we decided to change the optimizer for SGD to see how it performed.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/sgdtransloss.png)


### Experiment 4: Data Augmentation

From the beginning of the project we wanted to implement this classic ML technique to boost the model prediction. In spite of this technique be often used to reduce overfitting, we have not suffered such, but we wanted to test our net and see how the affected accuracy in validation and test. The transformations done to the images comprehend random horizontal flips and modifications to brightness, contrast, saturation and hue.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransdaloss.png)



### Experiment 5: Inverted Weight
Weights were added to the loss using the inverted frequency. Using the information obtained from train split, the inverted frequency was calculated for each class.
**Creo que alberto puede explicar mejor esta parte**
![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransdainvloss.png)

### Experiment 6: Weather Augmentation
In order for the network to prepare for varying road scenarios, in this experiment, it was trained while running the weather augmentation online to generate rain and snow. 
After running the data augmentation experiment and even though not having valuable results, we decided to include some realistic data augmentation. In our case, driving scenario, would be very helpful to add circumstances that drivers find on the daily. Of course, this should help the model to generalize better in exchange of a decreasing validation accuracy. 
The photos were added a layer of one of these elements (rain, snow, clouds, fog) *using python library imgaug*.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransweloss.png)

### Experiment 7: Deeplabv3
Finally, the entire project was run using the pre-w eighted Deep Lab model to compare the results to our U-net.
**Creo que alberto puede explicar mejor esta parte**

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamdeeplabloss.png)

### Experiment 8: Change Deeplabv3 optimizer


![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/sgddeeplabloss.png)


### Experiment 9: Change learning rate

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/sgd01deeplabloss.png)


### Experiment 10: Add weather data augmentation

** Experimento en marcha 12.04 13:50 **

### Metrics

Evaluating and comparing the experiments is a nuclear part of the scientific work and opens the path to adjust parameters and propose changes. For this project we defined several metrics to compare models and trainings

#### Accuracy
The main metric we used to evaluate the experiments if the accuracy of the model configuration. The model prediction accuracy  is calculated dividing the number of encerted pixels by the number of total pixels. However, there is a class that we are ignoring throughout the experiments and does not compute for the accuracy.
Next we show the accuracy comparison: 

*(mpl accuracy graph comparing valuable experiments)*


#### IoU per class
The previous metric is a generalization of how our model works overall. This next one gives emphasis on the nature of the data. Intersection over Union (Jaccard score) is an effective, well known metric for pixel classification in object segmentation problems. The IoU score for each class is a number between 0 and 1 calculated by dividing the intersection from the pixels in prediction and GT by the union of these two surfaces.

As we presented in the dataset statistics, we have a noticeable class imbalance, which ends up in an unbalanced IoU. The classes that appear the most in the dataset (pavement, sky) reach a higher IoU than the ones that appear very few times (signals, traffic lights)


| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 0.80 | 0.64 | 0.84 | 0.25 | 0.12 | 0.39 | 0.32 | 0.49 | **0.87** | 0.44 | 0.86 | 0.43 | 0.21 | 0.83 | 0.11 | 0.22 | **0.00** | 0.07 | 0.50 |


#### mIoU

The other metric that illuminated our grievous path through the fathomless darkness of semantic segmentation was the mean Intersection over Union. A mean calculation of every class IoU is used to measure how well is in classificating all the classes.

*(mIoU graph comparing valuable experiments)*


#### Validation results

| Optimizer (LR) | Model | Version | Configuration | Accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) |  UNet| Linear||82.25 | 40.1
|  |  | Bilinear 3x3/1||83.13 |41.7
|  |  | Bilinear 1x1|| 83.46 | 43.23
|  |  | Transpose| |83.64|44.01  
| SGD (0.001) | | Transpose|| 80.89|34.26 
| Adam (0.001) |  | Transpose|DA | 82.77| 41.33
|  |  | Transpose|DA & IF |75.14|35.09
|  |  | Transpose|Weather DA |81.28|38.41 
|  | Deeplabv3 | | |82.32|39.47 
| SGD (0.001) | Deeplabv3 | | | | 
| SGD (0.1) | Deeplabv3 | | |83.99|46.64 
|  | | |Weather DA | | 


#### Test results

The previous metrics were taken in the validation phase of our training. Concluding the experiment we test the model configuration with the test dataset. The results in this phase give us an overall understanding of the performance. 
*(excel table of the results)*

| Optimizer (LR) | Model | Version | Configuration | Accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) |  UNet| Linear| | 76.7| 38
|  |  | Bilinear 3x3/1| | 77.9| 40
|  |  | Bilinear 1x1| |78.5 | 42
|  |  | Transpose| | 78.3| 43
| SGD (0.001) | | Transpose| |75.3 | 31
| Adam (0.001) |  | Transpose|DA |77.3 | 39
|  |  | Transpose|DA & IF |70.16 |34 
|  |  | Transpose|Weather DA |75.72 | 35
|  | Deeplabv3 | | |76.86 | 39
| SGD (0.001) | Deeplabv3 | | | | 
| SGD (0.1) | Deeplabv3 | | |78.73| 46
|  | | |Weather DA | | 


### Other comparisons


We don't see any noticeable difference in the loss plot between these two versions
![loss331vs1](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/loss331vs1.png)

---

After all the "heavy" experiments, we decided to play a bit with the configuration, and swapped between common optimizers that might help us with better results after several failed attempts and give us some fresh air.
In the next figure we can see that after a similar start in the first 15 epochs, and penalized by the low start, SGD has a steeper mIoU curve and might need more epochs to reach Adam's performance in this metric.
![miouadamvssgd](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/miouadamvssgd.png)

---

![acctransvsda](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/acctransvsda.png)

There really is no noticeable improvement after adding this transformations. We expected it also in the test results, but we did not get there an improvement either:

| Configuration| Accuracy | mIoU 
|--|--| --|
| Normal |78.3  |43
| Data augmentation | 77.3  |39

---

Our last experiment was changing the learning rate of the optimizer. We did it on several configurations, both UNet and Deeplabv3 and both Adam and SGD, and here we can notice a real change.

![miou01vs0001](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/miou01vs0001.png)


## Conclusion

(we should decide what out conclusions for the project are and insert here)
![Linear timeline](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/linear.gif)
![Bilinear timeline](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/bilinear.gif)




## Future Work

 - Driveable Zone Segmentation
 - Pruning
 - Focal Loss

## References
