#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./md_img/training_data_hist.png "Training Data Distribution"
[image2]: ./md_img/testing_data_hist.png "Testing Data Distribution"
[image3]: ./md_img/validation_data_hist.png "Validation Data Distribution"
[image4]: ./md_img/all_class_show.png "Shows all classes"
[image5]: ./md_img/all_class_show_gray.png "Shows all classes gray"
[image6]: ./md_img/data_augment_example.png "Data Augment EX"
[image7]: ./md_img/training_result.png "Training Result"
[image7]: ./md_img/extra_img.png "Extra Images"
[image8]: ./md_img/predicition_new_img.png "New img prediction"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
The size of training set is 34799
* The size of the validation set is ?
The size of validation set is 4410
* The size of test set is ?
The size of testing set is 12630
* The shape of a traffic sign image is ?
The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is ?
43

####2. Include an exploratory visualization of the dataset.

The plots below show the distribution of each dataset.

![alt text][image1]
![alt text][image2]
![alt text][image3]

Below are pictures of each class.
![alt text][image4]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because according to Yann Lecun, color channels do not improve the performance of an image classifier and it only makes the data size big to hinder the preprocessing of the images. 

I also equalized the exposures because some of the images looked too dark.

Here are all classes after converting to gray scale and exposure equalization.
![alt text][image5]

As a last step, I normalized the image data because the classifier performance improves when the dataset has zero mean and unit variance. 

I decided to generate additional data because the original dataset seems to be very unevenly distributed. 

To add more data to the the data set, I generated more images for classes with less than 1000 images (chosen rather arbitrarily) so all the classes have at least 1000 images.

The technique I used are 
* Rotation (between -2 to 2 degrees)
* Zoom (between 0.95 to 1.05)
* Translation (0 to 2 pixels)

For each image generation, the parameters were chosen with uniform distribution. The parameters ranges are small because too much of transformation can cause the image to have different identity.

Here is an example of an original image and an augmented image:

![alt text][image6]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| sigmoid					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 				|
| sigmoid					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 120 neurons       									|
| sigmoid					|												|
| Fully connected		| 84 neurons       									|
| sigmoid					|												|
| Fully connected		| 43 neurons       									|
| sigmoid					|												|
| Softmax				|         									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The training parameters I used are
* learning_rate = 0.01
* keep_prob = 0.8
* n_epoch = 50
* batch_size = 100

For the learning rate, I started with 0.0001, but realized since the training dataset grew in size, I should increase the learning rate. For the keep probability for dropout, I started with 0.5 but since it took too long for the classifier to start making meaningful outputs, I increased it to balance time against over fitting. I originally tested with 100 epochs of training but around 50 epochs, the network barely improved performance. 

Below is the training result
![alt text][image7]

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 76.439%
* validation set accuracy of 95.714%
* test set accuracy of 93.658%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried using the inception model with very deep network.
* What were some problems with the initial architecture?
Since the netowrk was very complex, it took very long time to train. In order to tune the parameters, I had to iterate the network multiple times and I let the network run for at least 5 epochs to see if the network gets stuck or keeps improving. This process took too long for the inception network. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I just stuck with the LeNet structure and focused more on data pre processing because that seems to yield more performance improvement. Also, since LeNet structure is relatively simpler and shallower, it took much less time to run the iterative process of tuning hyper parameters. 
* Which parameters were tuned? How were they adjusted and why?
I tuned learning rate, batch size, number of epochs, and drop rate because these seem to affect the network the most. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layer works well because it is able to extract import features from the data. Dropout layer, which I used after each layer, helps the network from overfitting. 

If a well known architecture was chosen:
* What architecture was chosen? LeNet.
* Why did you believe it would be relevant to the traffic sign application? It is a classic image classifier. I thought it would work well with the traffic sign "images"
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Since the validation and testing dataset both achieve over 90% classification accuracy, the classifier must be working correctly because the network was never exposed to these datasets during training phase. The relatively low training dataset accuracy is probably due to dropout.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image7]

All the images were much higher resolution than the images I used to train the classifier. The images I downloaded were roughly around 300 x 300 (around 100x larger than the training images). The images were down sampled to match the 32 x 32 size restriction. during this process, the image lost a lot of pixels (information). Some images like class 21 and class 28 look very similar when the resolution is low. 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][image8]


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

True Class ------< 11 >------

[11: 0.5111]  [29: 0.3051]  [28: 0.0819]  [21: 0.0684]  [30: 0.0247]  


True Class ------< 13 >------

[13: 0.9998]  [14: 0.0002]  [ 5: 0.0000]  [38: 0.0000]  [35: 0.0000]  


True Class ------< 14 >------

[14: 0.5463]  [29: 0.1720]  [25: 0.0743]  [28: 0.0396]  [27: 0.0317]  


True Class ------< 21 >------

[28: 0.9918]  [29: 0.0031]  [25: 0.0021]  [30: 0.0014]  [24: 0.0011]  


True Class ------< 22 >------

[22: 0.9018]  [29: 0.0918]  [28: 0.0049]  [23: 0.0007]  [25: 0.0004]  


True Class ------< 28 >------

[28: 0.2239]  [ 5: 0.1612]  [13: 0.1129]  [40: 0.1007]  [ 3: 0.0837]  


True Class ------< 31 >------

[29: 0.3436]  [23: 0.2229]  [31: 0.1971]  [22: 0.1339]  [20: 0.0614]  


True Class ------< 38 >------

[38: 1.0000]  [ 5: 0.0000]  [12: 0.0000]  [40: 0.0000]  [ 2: 0.0000]  


True Class ------<  4 >------

[ 4: 0.9902]  [ 1: 0.0079]  [ 0: 0.0016]  [35: 0.0001]  [33: 0.0000] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


