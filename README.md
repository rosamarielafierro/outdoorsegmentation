
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
 - Authors: Alberto Masa, Fernando del Tíbet, Mariela, Sina Sadeghi
 - Institute: Universitat Politecnica De Cataluña

## Motivation

Our field of interest was driving-related topics. This includes a deeper understanding of image-video processing for understanding the context around the vehicle to improve safety.

## Proposal
*This is copied from the presentation. Maybe we can detail this with what we've done*

 - [x] Analyze the data provided in the selected dataset and adapt it to be used in a Semantic Segmentation network.
 - [x] Mitigate the class imbalance based on a better understanding of our data.
 - [x] Learn how to do a transfer learning from the previous task to another one, for instance, detecting the drivable area.
 - [x] Reproduce a semantic segmentation network described in the U-Net paper from scratch.
 - [x] Apply data augmentation, generate different kinds of weather such as fog, rain, snowflakes.

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

![Unet Model Diagram](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Unet%20-%20Basic.png)

This allows the network to reduce spatial information while increasing feature information on its way down, and reduce feature information while increasing station information on the way up, leading to highly efficient image processing. The U-net class is defined as follows,

(Insert Unet class code screenshot)

### Optimizer
(help)

### Concatenation Layer
Since the U-net downsamples the feature information in the first half of the network, there is a risk of loosing valuable information. To overcome this, we concatenated all the feature maps in the decoding layers with the feature maps from the encoding layers. This assures that any information learned in the ignitions layers will be retained throughout the network. 

![Concat Layers](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Unet%20-%20Concat.png)

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
Throughout the process of advancing the network, multiple experiments were conducted in order to track progress. What drove us during the whole process of experimentation was evidence :eyes:. What we searched over and over was improvement :sunrise_over_mountains:. But we didn't find it :trollface:, at least how we expected. We expected to see results as we have been seeing in the labs and in the ML books, but maybe we had a stroke of reality. This is the nature of science and engineering :smirk:. We stuck with the configuration that gave us better results until the moment and build on top of it.
The first 6 experiments consists of adjusting configurations for the network itself and adding ML techniques. For the last two experiments we wanted to play a little bit and test the theoretical part we learned in class: we changed the optimizer for our network and an hyperparameter, the learning rate.

**Fer: pongo algunas gráficas y tablas y después con Albert&co vemos cual son más significativas **

### Experiment 1: Linear UNet
The first experiment was intended to act as a base-line for the all future experiments. It consisted of a linear version (removing the concatenations) of the UNet using Adam optimizer. This lightweight version works for us to see a quick segmentation result that is easy to understand and easy to be improved by adding components to the configuration. On the other hand it gives very little precision to the prediction. The architecture is presented in the figure #1 and the method to upsample is *torch.nn.Upsample*. 

(statistics image)

![Linear timeline](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/linear.gif)

### Experiment 2: Concat UNet
For the second experiments, we improved the network to embrace the concatenations defined in the canonical net. Also, we implemented another way to upsample the encoded data in the net through Transposed convolutions.

![Bilinear timeline](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/bilinear.gif)

#### Transpose Kernel 3 Padding 1 vs Transpose Kernel 1
*Explicar porqué hemos hecho esto*


    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding= 1))
  ---

    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1))

We don't see any noticeable difference in the loss plot between these two versions
![loss331vs1](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/loss331vs1.png)

### Experiment 3: Data Augmentation

From the beginning of the project we wanted to implement this classic ML technique to boost the model prediction. In spite of this technique be often used to reduce overfitting, we have not suffered such, but we wanted to test our net and see how the affected accuracy in validation and test. The transformations done to the images comprehend random horizontal flips and modifications to brightness, contrast, saturation and hue.

This is the result:
![acctransvsda](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/acctransvsda.png)

There really is no noticeable improvement after adding this transformations. We expected it also in the test results, but we did not get there an improvement either:

| Configuration| Accuracy | mIoU 
|--|--| --|
| Normal |78.3  |43
| Data augmentation | 77.3  |39


### Experiment 4: Inverted Weight
Weights were added to the loss using the inverted frequency. Using the information obtained from train split, the inverted frequency was calculated for each class.
**Creo que alberto puede explicar mejor esta parte**
(statistics image)

### Experiment 5: Weather Augmentation
In order for the network to prepare for varying road scenarios, in this experiment, it was trained while running the weather augmentation online to generate rain and snow. 
After running the data augmentation experiment and even though not having valuable results, we decided to include some realistic data augmentation. In our case, driving scenario, would be very helpful to add circumstances that drivers find on the daily. Of course, this should help the model to generalize better in exchange of a decreasing validation accuracy. 
The photos were added a layer of one of these elements (rain, snow, clouds, fog) *using python library imgaug*.

**Cuando hagamos el experimento online bien enchufarle una gráfica**

### Experiment 6: Deep Lab
Finally, the entire project was run using the pre-w eighted Deep Lab model to compare the results to our U-net.
**Creo que alberto puede explicar mejor esta parte**

*(maybe architecture or some other graph)*

### Experiment 7: Change optimizer

After all the "heavy" experiments, we decided to play a bit with the configuration, and swapped between common optimizers that might help us with better results after several failed attempts and give us some fresh air.
In the next figure we can see that after a similar start in the first 15 epochs, and penalized by the low start, SGD has a steeper mIoU curve and might need more epochs to reach Adam's performance in this metric.

![miouadamvssgd](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/miouadamvssgd.png)

### Experiment 8: Change learning rate

Our last experiment was changing the learning rate of the optimizer. We did it on several configurations, both UNet and Deeplabv3 and both Adam and SGD, and here we can notice a real change.

![miou01vs0001](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/miou01vs0001.png)


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
|  | Deeplabv3 | | |78.73| 46
|  | Deeplabv3 | |Weather DA | | 

## Conclusion

(we should decide what out conclusions for the project are and insert here)




## Future Work

 - Driveable Zone Segmentation
 - Pruning
 - Focal Loss

## References