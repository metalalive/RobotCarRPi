# Raspberry PI X Robotic car X Tensorflow

Here's description about how I build a robotic car using Raspberry PI with car kits and camera, I'll also describe about how I program the car, build/train/validate my neural network for image processing using **Tensorflow C++ API**, I also managed to get 85% accurancy when predicting images of test sets using the trained neural network model.

### Hardware Components
* Raspberry PI 3B+
* USB webcam
* car chassis kit
* L298N motor controller
* DC motors (3-6 Volt.) x 4
* 1.2 Volt. battery x 6

### Software Components
* Raspbian 9 (Stretch) with GCC 6.3.0
* Build shared libraries from sources for our C++ application, this includes :
  * opencv 4.0.0
  * Tensorflow 1.12
  * protobuf 3.6.0
  
  please read [How to build shared libraries](build_essential_libraries.md) for detail.


### The track
I decided to build my own track since I couldn't find any existing track in my hometown, it is about 7-8 meters x 4 meters, black-lane centered track, due to illumination changes and material of the floor, it could be challenging to predict lane line using traditional computer vision approach.

<img src="track1.jpg" width="383" height="474" class="center" />

### The Car

<img src="robotCarPI.jpg" width="416" height="312" class="center" />

Here is GPIO pins wiring to L298N controllers



#### Image collector
I also wrote a tool in Python for following objectives:
* recording video of my lane lines 
* labeling frames from recorded videos.

For data collection, we mounted the camera on chassis, manually moved the chassis with respect to few driving situations :
* the car perfectly follows the lane line
* overshoot the lane a little bit
* overshoot with sharp angle

I ended up with 10 videos, and ~30000 image examples with labels.

To create labeled dataset, we make use of open CV to load each frame from recorded videos to the GUI window, use mouse to click on the expected centroid of the lane line of each frame, press any key on keyboard to switch to next frame, then our tool will automatically :
* resize the frame to smaller 160 x 120 frame
* saves the frame to image file
* normalize the point ((x,y) in image coordinator) we clicked for the frame, the normalized (x,y) value will be in the range of -1 to 1.
* add label information (normalized point & the path of the image ) to csv file.



### Why Tensorflow C++ ?
My objective here is to build C++ based neural network application, Tensorflow C++ API seems like the easiest way to do so. A lot of similar examples on the internet descibe that you can get better resulting/validation result with Tensorflow python API, since many exotic / convenient operations have been implemented there (also many optimizers available), however in my case, I found that TensorFlow python API only improves 3-4% accurancy (~89%) than its C++ API (~85%) with the same dataset, even I applied optimizer at the python side and convolutional layer at the both side, I couldn't get that much improvement. So I still choose Tensorflow C++ API.

### building model

### hyperparameters

### train the model

### save the mode1 to protobuf text file

### save the parameter matrices to checkpoint files

### load the trained model on Raspberry Pi




