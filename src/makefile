
SRC_FOLDER?=
DST_FOLDER?=

# either host or rpi
TARGET?=host 

# usage :
# make xxx DEFINE_CC_PARAMS="param1 param2 param3 ..."
# it will translate to 
#     -Dparam1  -Dparam2  -Dparam3
#     then put it into compile command line
DEFINE_CC_PARAMS  ?=  

CC := g++ 
LD := g++


HEADER_RPI:= -I/usr/local/include/opencv4 \
             -I../include/tf/1.11/tensorflow/contrib/makefile/downloads/eigen \
             -I../include/tf/1.11/tensorflow/contrib/makefile/downloads/absl \
             -I../include/tf/1.12/bazel-genfiles \
             -I../include/tf/1.12/third_party \
             -I../include/tf/1.12/ 

HEADER_UBUNTU:= -I/usr/local/include/opencv2 \
               -I../include/tf/1.11/tensorflow/contrib/makefile/downloads/eigen \
               -I../include/tf/1.11/tensorflow/contrib/makefile/downloads/absl \
               -I../include/tf/1.11/bazel-genfiles \
               -I../include/tf/1.11/third_party \
               -I../include/tf/1.11/


LIB_RPI:=    -L../lib/protobuf/3.6.0/rpi/3         -L../lib/tensorflow/1.12/rpi/3 \
    -Wl,-rpath=../lib/protobuf/3.6.0/rpi/3  -Wl,-rpath=../lib/tensorflow/1.12/rpi/3 \
    -lprotobuf  -ltensorflow_cc 

LIB_UBUNTU:= -L../lib/protobuf/3.6.0/ubuntu/14.04  -L../lib/tensorflow/1.11/ubuntu/14.04 \
    -Wl,-rpath=../lib/protobuf/3.6.0/ubuntu/14.04  -Wl,-rpath=../lib/tensorflow/1.11/ubuntu/14.04 \
    -lprotobuf  -ltensorflow_cc -ltensorflow_framework


INCLUDE_HEADER := -I../include  -I.  -I../include/protobuf/src 

LINK_LIB := -lpthread  -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_photo -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core 



ifeq ($(TARGET), rpi)
DEFINE_CC_PARAMS += HOST_OS_RPI
INCLUDE_HEADER += $(HEADER_RPI)
LINK_LIB       += $(LIB_RPI)
else
INCLUDE_HEADER += $(HEADER_UBUNTU)
LINK_LIB       += $(LIB_UBUNTU)
endif



PREFIX_CC_DEFINE_PARAMS  =  $(foreach item, $(DEFINE_CC_PARAMS),  -D$(item) )

CC_OPTIONS = -Wall  -nostartfiles -c -g  -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1 -pthread \
              -Wfatal-errors




%.o: %.cc
	$(CC)  $(INCLUDE_HEADER) $< -o $@  $(CC_OPTIONS) $(PREFIX_CC_DEFINE_PARAMS)

%.out: %.o  GPIOcontroller.o  dataset_handler.o  models.o 
	$(LD)  $^ -o $@  $(LINK_LIB)


run:
	@sudo modprobe bcm
	@./main.out  /dev/video0  ../models/models.pbtxt  \
           ../models/trained_param_checkpoints__acc_87p.bin  470


copy_tf_headers: $(SRC_FOLDER)
	@rm -rf $(DST_FOLDER)
	@mkdir -p $(DST_FOLDER)/tensorflow
	@cp -r $(SRC_FOLDER)/bazel-genfiles  $(SRC_FOLDER)/third_party  $(DST_FOLDER)/
	@cp -r $(SRC_FOLDER)/tensorflow/core $(SRC_FOLDER)/tensorflow/cc  $(DST_FOLDER)/tensorflow/
	@make cleanup_files REGEX_INPUT=".*[.][^h]*" DST_FOLDER=$(DST_FOLDER)

cleanup_files: 
	@echo "#!/usr/bin/env bash" > ./__tmp.sh;
	@echo "rm -f  $(shell find $(DST_FOLDER)/ -regextype sed -regex "$(REGEX_INPUT)" -type f)" >> ./__tmp.sh;
	@chmod 700 ./__tmp.sh
	@./__tmp.sh
	@rm -rf ./__tmp.sh

clean:
	@rm -rf  *.o *.out 


help:
	@echo "----- help document -----"
	@echo ""
	@echo "g++ version 6.3.0 is suggested in Raspbain Stretch. "
	@echo ""
	@echo "make train.out "
	@echo ""
	@echo "make lane_prediction_test.out TARGET=rpi "
	@echo ""
	@echo "make copy_tf_headers  SRC_FOLDER=xxx  DST_FOLDER=xoxoxo "
	@echo ""
	@echo "./train.out  cluster_labels_60_grps   ./  0.00004  0.000001  0.0011  0.0625  0.1245  32 8  800 "
	@echo ""
	@echo "./lane_prediction_test.out  cluster_labels3_30_grps  . "
	@echo ""
	@echo ""


#ifeq ($(MAKECMDGOALS), copy_tf_headers)
#endif
### $(shell pkg-config --cflags opencv) 
### $(shell pkg-config --libs   opencv)
