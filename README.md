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
7-8 meters x 4 meters, black-lane centered track, due to illumination changes and material of the floor, it could be challenging to predict lane line using traditional computer vision approach.


### The Car


Here is GPIO pins wiring to L298N controllers


### Why Tensorflow C++ ?
My objective here is to build C++ based neural network application, to do this, Tensorflow C++ seems like the easiest way, a lot of similar examples on the internet descibe that you could train better model with Tensorflow python API, since many exotics / convenient operations have been implemented there (also many optimizers available), however in my case, I found that TF python API only gives 3-4% impprovement (~89%) than TF C++ API (~85%), with the same dataset, even I applied optimizer at the python side and convolutional layer at the both side, I couldn't that much improvement.

#### Image collector
since I couldn't find any existing track, also makerspace is super far from my hometown, I decided to build my own one. I also wrote python code to collect images of my lane lines with respect to a couple of situations :
