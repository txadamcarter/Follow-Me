# RoboND 'Follow Me' Project Writeup

[image1]: ./misc_writeup_images/transposed-conv.png
[image2]: ./misc_writeup_images/SkipConnections.png
[image3]: ./misc_writeup_images/One_By.png
[image4]: ./misc_writeup_images/comp_graph.png


## Explain each layer of the network and the role that it plays. 
## Demonstrate benefites/drawbacks of different architectures, justifying current choice of architecture and hyperparameters using data. Provide graphs, tables, diagrams as reference.

The 'winning' architecture that I ended up with was found through exhaustive experimentation with layer depths and to a lesser extent, hyperparameters. Using the rather shallow data set provided to train with (no supplemental data was used due to the sim crashing) made the situation into a game of finding the right number of parameters to train. Too many and the system would overfit immediately. This was clearly evident from very low training loss in the face of high validation loss. Too few parameters and the system would fail to learn altogether. The slice of workable system widths proved narrow. The take-home is that more data is desperately needed in order to make a better system. This system has four encoder and four decoder layers. The encoder layers are separable convolutions. The decoder layers each have a bilinear upsampling step, a concatenation step to form the skip connection and two 'normal' conv2D layers prior to output. There is a 1x1 convolution separating encoder from decoder.


![Computational Graph][image4]


Fully Convolutional Networks (FCN) are convolutional networks with an 'encoder' section and 'decoder' section separated by one or more 1x1 convolutions rather than fully connected layers. 

### Encoder 

The encoder portion of the network is composed of a number of separable convolutions. These are normal convolutions over each channel of an input layer followed by 1x1 convolution prior to output. This allows location data to be preserved and reduces parameters which helps prevent overfit and improves computational efficiency. Keras provides an optimized version:

output = SeparableConv2DKeras(filters, kernel_size, strides, padding, activation)(input)

This is included in the function:

separable_conv2d_batchnorm()

#### A Note on Batch Normalization
Batch normalization regularizes hidden unit's activations and speeds up learning. It reduces the amount that values shift when undergoing covariate shift and therefore makes learning easier in a similar way to what happens when we normalize inputs to have 0 mean and variance 1. We are now doing this for the inputs to each layer in the network rather than just to the original inputs to the network. This makes weights found later in a network more robust to changes in earlier layers and stabilizes learning somewhat. No matter what values these parameters may take on, they will always have the same mean and variance which reduces the problem of highly shifting input values destabilizing the system. Scaling each mini batch by it's own computed mean/variance adds some noise to the values. Similar to dropout, this adds some noise to that layer's activations which has a slight regularizing effect.

This batch-norm information is partly from Andrew Ng's Coursera course.

### 1x1 Convolution

The next major section of the network is the 1x1 convolution that separates the encoder from the decoder. Properties of 1x1 convolutions are discussed separately below.

### Decoder

Each decoder block section has three parts: 
1) Bilinear upsampling layer. This reverses the dimensional changes induced by the corresponding layer in the encoder.
2) Concatenation step. This connects the upsampled layer to a larger layer to form the skip connection.
3) Separable convolutional layers to extract more spatial information


## Fully Convolutional Networks: Per-Pixel Classification with Retention of Spatial Data

 The 1x1 convolutional architecture, along with skip connections and upsampling, allows for pixel-by-pixel classification of an entire image regardless of input size while retaining spatial and class information for each pixel. This is referred to as semantic segmentation. It is interesting to note that a network can have more than one 'decoder' section. For instance, one decoder can be used for semantic segmentation while another is used for depth classification.

Compared to traditional Convolutional Networks, a Fully Convolutional Network has the ability to not only classify objects as to type but also retain the spatial information about that object's location in the frame. This is accomplished via three main adaptations to the network architecture:

1) 1x1 Convolutions 
2) Skip Connections 
3) Transposed Convolutional Layers (Upsampling) 

I will discuss each of these presently.


## 1x1 Convolutions Preserve Spatial Information

![1x1][image3]

From paper titled 'Network in Network'

one_by = conv2d_batchnorm(enc_4, filters=64, kernel_size=1, strides=1)

A 1x1 convolution is simply a normal convolutional layer but with filter dimensions of one pixel in each direction, a stride of one and 'same' or zero padding. They are substituted for fully connected layers so as to avoid the loss of information that comes along with 'flattening' a 4-D tensor (standard output of a convolutional layer) down to 2-D to be fed into a fully connected layer. After the 1x1 convolves over the previous layer's output in it's full 4-D glory, the information pertaining to the target pixel's location relative to the pixels in it's vicinity is passed along to the next layer rather than being lost. Replacement of fully connected layers with 1x1 convolutions also can be used to reduce computational complexity by reducing dimensionality. They come with the added benefit of making the network invariant to input image size during testing.


## Skip Connections Combine Small and Large Scale Perspectives of the Data

![Skip Connections][image2]

From Leonardo Aroujo dos Santos GitBook

By narrowly looking at small areas of the image when we convolve, we lose some of the 'big picture' relationships between the target and it's surroundings. Information is lost in this way even if the layer dimensionality is re-established using upsampling. To avoid this we concatenate (join by element-wise addition) the output of an early layer with that of a later non-adjacent layer. The early layer contains information spread out over a larger window size compared to the more narrowly focused later layer. By combining these two, we can present later layers with information rich in both fine detail and larger scale spatial information. The network is thereby enabled to make more precise segmentation decisions.


## Transposed Convolutions Upsample Data to Keep Input/Output Dimensionality Congruent

![Transposed-Convolution][image1]
From Udacity

Behind every layer of a neural network there is a linear algebra function called a matrix-multiply. In order to multiply two matrices, their dimensions must be compatible. It is this underlying requirement that compels us to manage the dimensionality of our layers. After the downsampling associated with convolution, we must upsample in order to retain dimensional congruency.
The upsampling function of a transposed convolution is accomplished via manipulation of the kernel size, strides and padding. In the example above we have a 2x2 input, kernel size of three and stride of two. This will result in an output dimension of 4x4. Through careful manipulation of these variables, we can increase or decrease dimensionality at will, providing the opportunity to produce output at the same size as the input. Or train pixels to pixels in other words. In our implementation we used bilinear upsampling rather than transposed convolutions to achieve the same result. 


## Explanation of hyperparameters and how they were determined

### Epochs

The number of Epochs used to train a neural network is in part dependent upon the size of the training data set relative to the number of parameters needing to be trained in the network. A network with many parameters trained on a small dataset is likely to overfit easily if too many epochs are run. A smaller network trained on a large dataset may be more resistant to overfit until very many epochs have been run. This does not speak to the number of epochs best used to achieve low error scores, but rather how susceptible it may be to overfit. In either case, if too many epochs are run, the network will eventually overfit on the training data, especially if it is not a large training dataset. This results in very low loss on the training set (strong learning) but poor validation performance (weak prediction) when the trained network is tested on data it has never seen. Therefore the number of epochs is chosen so as to approach convergence of the training and validation error rates toward a minimum without going past that point. This is referred to as early stopping.

### Learning Rate

In any given neural network the learning rate refers to the portion of each backpropagated error derivative that is used to adjust the weight. We forward propagate a batch of data through the network, calculate the error at the end, run backpropagation to spread that error properly over the network so weights can be appropriately adjusted to reduce error then repeat. Each cycle we choose a small portion of that calculated error to apply a small motion to the weights in the right direction. It is important that this be a small portion of the calculated error so as to minimize overfitting on any given datum. In this way the error function can be descended more smoothly and without wild swings. Typical learning rates lie in the realm of 0.01 to 0.000010. There are many situations in which lowering the learning rate can help smooth training. I chose a fast learning rate of 0.01 due to this particular system's robustness. Lower learning rates did not improve learning in this application. 

### Batch Size

Batch Size tuning is sometimes like learning rate tuning in that it is very situation dependent. The memory size available on the system being used for training influences optimum batch size directly. So this parameter must be optimized by running training with different batch sizes and tuning the network appropriately. The optimum for my system turned out to be around 32-64.


## Discuss image manipulation/encoding/decoding including when, why and what problems are encountered.

In any given image there is a lot of information. Each pixel has:

 1) a certain color (usually encoded as a scalar between 0-255 in red, green and blue channels)
 2) an x,y coordinate location within the frame
 3) perhaps a 'z' depth value
 4) a certain set of neighbor pixels, all with their own values for these characteristics. 

 It is helpful to recognize this and relate it to the problem you are attempting to solve with the network you are training the data on. Perhaps by selectively looking at one channel or selectively deleting the contribution of one channel classification is easier to accomplish in a certain dimension. In this application the 'hero' character whom the 'Follow Me' function should follow has a distinct appearance relative to the other characters in the world. She has a red shirt on which makes her stand out. Filters such as these occur naturally in many images and help distinguish objects when they are otherwise largely similar. If no distinct differences in clothing color are apparent, then another feature can potentially be used to classify and distinguish amongst similar characters. Body shape, hair, and virtually any other visible feature can potentially be useful. When we encode and decode our image data we must be cognizant of any effects we may have on the data when we apply filters in certain stages. Sometimes data can be recovered, other times it is lost following filtration. Often we are in charge of which is which.


## Demonstrate understanding of the limitations of a NN given different data scenarios and demands. Would it work as well following a different object? In a different environment? What changes would help it perform better in those conditions?

More data is always a good thing. When I tried to collect more training data from the sim, it crashed so frequently that I found it impossible to use. I had heard on Slack that it is possible to train the network to a passing score using only the training data supplied by Udacity. So I figured I should be able to do that. Slightly different flavor challenge but cool nonetheless. This proved difficult however and the main lesson I learned was that it is easy to overfit on a small dataset. I think I went too deep with my network's size early on and had difficulty due to that. When I finally learned to pare the network down to fit the small data size, I got better results. 

With it's current small architecture I have to assume the network is not operating on an abundance of parameters. It therefore should not be great at making complex inferences and I would predict that following objects dramatically different than our target may be difficult. The same should likely be true concerning the environment. Although varied, the provided sim environment is still rather simple compared to the astounding variety in nature. Put my network in a more complex environment and I would not guarantee great performance. However, it should provide a reasonable starting point for a similar but beefier network to be trained on more real-world data. By beefier I mean deeper with regard to number of layers and possibly dimensionally deeper also with regard to number of filters per layer. This would increase parameters and therefore make training more difficult but should result in a more robust predictive system.

To summarize, I would not expect this system to perform well in a situation where the hero became a dog, cat or other object substantially different from the hero we trained on. The specific visible features (size, shape, color) that we train the system on are the only features it learns. So training a system on one set of data and expecting it to perform well when tested on very different data is not an intelligent approach. The system architecture, hyperparameters etc can almost always be tweaked enough to support learning, as I have demonstrated here. A much better approach, however, is to set up a system with a wide variety of data to learn from. The parameters learned by the system with robust data will be more robust and able to make effective predictions in a wider range of scenarios.

If I were to add a certain type of data, I would add more images containing the target (and non target characters) while the drone is on patrol. It seems from the appearance of the evaluation images that performance in those situations is poor. The network often misidentifies the target as a non-target character when the target is not directly in front of the camera.


## Future Enhancements

This model could be improved through addition of more data. The more examples of the hero and non-hero characters that the system is exposed to, the better it will get at differentiating hero from non-hero. Ideally I would supply the system with images taken from every concievable angle and distance from the camera, under all concievable lighting conditions, shadows, partial obscurity by environmental objects, overlapping with other characters etc. The more labelled instances of target vs non-target that it sees, the more likely it will be to properly differentiate those things under testing. The specific architecture needed to learn from that data would have to be found through experimentation. I think I can say that it would need more parameters than my system (thus more depth) in order to understand the richer training data but the specifics would need to be determined experimentally.


