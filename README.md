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


### The Car


Here is GPIO pins wiring to L298N controllers



#### Image collector
I also wrote a tool using Python for following objective:
* recording video of my lane lines 
* label frames from each recorded video, to do so we load each frame in the GUI window, use mouse to click on the expected centroid of the lane line of current frame, then our tool automatically saves the frame to image file, and saves label information ((x,y) in image coordinator) to csv file.

For data collection, we mounted the camera on chassis, manually moved the chassis with respect to a couple of situations :
* the car perfectly follows the lane line
* overshoot the lane a little bit
* overshoot with sharp angle

I ended up with 10 videos, and ~30000 image examples with labels


### Why Tensorflow C++ ?
My objective here is to build C++ based neural network application, Tensorflow C++ API seems like the easiest way to do so. A lot of similar examples on the internet descibe that you can get better resulting/validation result with Tensorflow python API, since many exotic / convenient operations have been implemented there (also many optimizers available), however in my case, I found that TensorFlow python API only improves 3-4% accurancy (~89%) than its C++ API (~85%) with the same dataset, even I applied optimizer at the python side and convolutional layer at the both side, I couldn't get that much improvement. So I still choose Tensorflow C++ API.



