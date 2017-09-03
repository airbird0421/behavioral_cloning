#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/counter-clock.jpg "Counter Clock Driving"
[image3]: ./examples/flip.jpg "Flipped Image"
[image4]: ./examples/left.jpg "Left Camera Image"
[image5]: ./examples/right.jpg "Right Camera Image

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 for the final demo video

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 85-94) 

Each convolution layer is followed by a RELU layer to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 72). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 97-99). By monitoring MSE output of both training sets and validation sets, I didn't see obvious overfitting because they are much close to each other. So I didn't use dropout layers in my model. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and counter-clock wise driving. I didn't specifically try recovering driving because I like the idea of left and right camera. I tried to use the left and right camera data to achieve effects of recovering driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to get a simple framework to work first, and then try more advanced models and optimize it.

My first step was to only use a flatten layer, which was simply to make sure the whole process works, including data reading from files, train-test splitting, model saving, and running the saved model in autonomous mode. In this step, I just used the sample data.

Then I added those basic, but necessary data preprocessing and augmenting techniques, including a lambda layer to normalizing the data, a cropping layer to remove the top and bottom parts of the images, and the augmentation by flipping the images. I also enabled generator because I knew later I would have much more data to process which would otherwise require a lot of memory.

I had a basic framework, or pipeline, with which, I could formally start my training work. I first generated my own training data, including two laps of center driving, and one lap of counter clock wise driving. I used LeNet architecture because it's a well known image recognition related architecture which should at least work as a start point. 

With these, I found the car could autonomously drive on the first part of track one, though it always kept to the left side of the road. I thought it was because I didn't have recovering driving data. I then enabled the left and right camera data, and tried to tune the correction angle. After some efforts, I was able to find a value, with which, on the first part of track one, the car could driver elegently.

Then I tried a more advanced network, the Nvidia model, just to see if it can bring any obvious difference. The result was that it didn't give me apparent improvement, but it didn't make things worse either. So I decided to keep with that model.

Since the car didn't work well on those sharp turns, I just captured more data on those two sharp turns. The added data greatly improved the performance on those difficult turns, but it also had some side effects on the other part of the track.

I thought maybe I had too much data on the turns now, also since I saw the power of more data, I just captured more data on the first part of the track so that the samples were not severely biased to one part. With these data, the car for the first time was able to drive through the whole track, though not perfectly, but it didn't drive off the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. At the beginning, I didn't pay much attention to overfitting. But in the last fine tuning stage, I tried to see whether overfitting was a problem. But luckily, it was not since the MSE of both training sets and validation sets were close. Though, I did try to add dropout layers but didn't see obvious difference. So I just removed them.

After capturing more data on the spots that the car didn't drive perfectly, in the end of the process, the car is able to drive autonomously around the track while keeping inside the lanes.

####2. Final Model Architecture

I just used the Nvidia model (model.py lines 85-94). It is a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 70x320x3 cropped and normalized image   							| 
| Convolution 5x5     	| 2x2 stride, output depth 24 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, outputs depth 36			|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, output depth 48 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, outputs depth 64			|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, outputs depth 64			|
| RELU					|												|
| Fully connected		| output 100        									|
| Fully connected		| output 50        									|
| Fully connected		| output 10        									|
| Fully connected		| output 1        									|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center_driving][image1]

I then recorded one lap of counter clock wise driving. Here's an example image of counter clock wise driving:

![counter_clock_driving][image2]

I didn't record the vehicle recovering from the left side and right sides of the road back to center but I used the left and right camera data to achieve the same effect, so that the vehicle would learn to go back to the center of the road when it is too close to the edge. These images shows the view of the left and right camera:

![left camera][image4]
![right camera][image5]

I also recorded more driving data on difficult spots where the car tended to drive off the road.

To augment the data set, I also flipped images and angles thinking that this would help the model to generalize better since most turns on track one are left turns. With flipped images, the model should be able to generalize to right turns. For example, here is an image that has been flipped:

![flipped_image][image3]

After the collection process, I had 40884 number of data points. The preprocessing includes normalization and cropping.


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the ever decreasing MSE. I used an adam optimizer so that manually training the learning rate wasn't necessary.
