# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
* pip install prettytable


[//]: # (Image References)
[image0]: ./output_images/Statistics.png "Statistics"
[image1]: ./output_images/Visualization.png "Visualization"
[image2]: ./output_images/Grayscaling.png "Grayscaling"
[image3]: ./output_images/data_augmention.png "Data augmention"
[image4]: ./extra/01.png "Traffic Sign 1"
[image5]: ./extra/02.png "Traffic Sign 2"
[image6]: ./extra/03.png "Traffic Sign 3"
[image7]: ./extra/04.png "Traffic Sign 4"
[image8]: ./extra/05.png "Traffic Sign 5"
[image9]: ./extra/06.png "Traffic Sign 6"
[image10]: ./extra/07.png "Traffic Sign 7"
[image11]: ./extra/08.png "Traffic Sign 8"
[image12]: ./extra/09.png "Traffic Sign 9"
[image13]: ./extra/10.png "Traffic Sign 10"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README
writeup.md



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image0]

### Design and Test a Model Architecture

#### 1. Describe how i preprocessed the image data. What techniques were chosen and why i choosed these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because there is only one channel for grayscale images, reduce data complexity

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it can adjusted data distribution,accelerated model training.

I decided to generate additional data because it can increase the size of the training set, improving model generalization ability.

To add more data to the the data set, I used the following techniques:
    transform_center = transf.SimilarityTransform(translation=-center_shift)
    transform_uncenter = transf.SimilarityTransform(translation=center_shift)
    
    transform_aug = transf.AffineTransform(rotation=np.deg2rad(angle_value),
                                          scale=(1/scaleY,1/scaleX),
                                          translation = (translationY,translationX))

    new_img = transf.warp(img,full_tranform,preserve_range=True) 


Here is an example of an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what my final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x80 	|
| RELU					|								
| Dropout	      	| 				



| Convolution 3x3	    | outputs 32x32x120   			|
| RELU					|								
| Max pooling	      	| 2x2 stride,  outputs 16x16x120 
| Dropout	      	| 	



| Convolution 4x4	    | outputs 16x16x180   			
| RELU					|								
| Dropout	      	| 

 

| Convolution 3x3	    | outputs 16x16x200   			
| RELU					|								
| Max pooling	      	| 2x2 stride,  outputs 8x8x200 
| Dropout	      	| 


| Convolution 3x3	    | outputs 8x8x200   			
| RELU					|								
| Max pooling	      	| 2x2 stride,  outputs 4x4x200 
| Dropout	

| Fully connected		| outputs 80 
| Fully connected		| outputs 80
| Fully connected		| outputs 43      				
| Softmax				|         						
 


#### 3. Describe how I trained my model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer (tf.train.AdamOptimizer(learning_rate=rate)), 
EPOCHS = 25
BATCH_SIZE = 200
learning_rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. My approach may have been an iterative process, in which case, outline the steps i took to get to the final solution and why i chose those steps. Perhaps my solution involved an already well known implementation or architecture. In this case, discuss why i think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9993
* validation set accuracy of 0.9943
* test set accuracy of 0.9632

EPOCH25...
Train accuracy:0.9993|Validation Accuracy=0.9943
Train loss:0.02554|Validation loss = 0.04450

Test Accuracy=0.9632


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Turn left ahead     	| Turn left ahead 			|
| Yield					| Yield											|
| Speed limit 30 km/h	      		| Speed limit 30 km/h					 				|
| Keep left			| Keep left    					|




#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 44th cell of the Ipython notebook.



#### 1. Discuss the visual output of my trained network's feature maps. What characteristics did the neural network use to make classifications?
shape, edge,texture,colour


