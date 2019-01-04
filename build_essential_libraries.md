To run the lane detection application on Raspberry Pi (3b+, in my case), we need:

##### OpenCV c++ library (v3.3.0 or 4.0.0): 
see installation guidiance [here](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/), it's ok to skip python part since we won't run python code on target Raspberry Pi.


##### Tensorflow C++ library (v1.11 or 1.12): 
the hardest part is right here, since there is no convenient package for C++ developers ...
###### Build Bazel from source (unfortunately no other easier way to doing this)
we managed to build bazel 0.17.0 for tensorflow v1.11, and build bazel 0.19.2 for tensorflow v1.12,
download bazel source from [HERE](https://github.com/bazelbuild/bazel/releases)
in command-line interface, you have:
```
wget https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel-0.19.2-dist.zip
unzip -qq bazel-0.19.2-dist.zip -d bazel-0.19.2
cd bazel-0.19.2
sudo chmod u+w ./* -R
```

the environment variable will be referenced when building bazel binary...
set java heap to 1 GB to avoid Java out-of-memory exception turns up
```
export BAZEL_JAVAC_OPT="-J-Xmx1g "
```
then 
```
./compile.sh
```
after compilation finished, copy the binary to :
```
sudo cp output/bazel /usr/local/bin/bazel
```

I was hitting compile error when trying up-to-date Bazel 0.21.0, therefore I recommend bazel 0.19.2 for tensorflow 1.12


###### use compiled Bazel binary file to build tensorflow C++ API from source
download tensorflow 1.12 source, extract it, switch to the folder, do the followings :
```
./configure
```
manually comment off the options in .bazelrc (see [issue #24416](https://github.com/tensorflow/tensorflow/pull/24416))
```
build --define=tensorflow_mkldnn_contraction_kernel=0
```
then start building (this could take several hours)
```
bazel build -c opt --copt="-mfpu=neon-vfpv4" --copt="-funsafe-math-optimizations" --copt="-ftree-vectorize" --copt="-fomit-frame-pointer" --local_resources 1024,1.0,1.0 --verbose_failures //tensorflow:libtensorflow_cc.so
```

###### download & build correct version of protobuf library

##### reference
https://gist.github.com/EKami/9869ae6347f68c592c5b5cd181a3b205
