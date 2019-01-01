
CC := g++ 
LD := g++





INCLUDE_HEADER := \
                  -I../include \
                  -I.  -I/usr/local/include/opencv2 \
                  -I../include/protobuf/src \
                  -I../include/tf/1.11/bazel-genfiles \
                  -I../include/tf/1.11/third_party \
                  -I../include/tf/1.11/tensorflow/contrib/makefile/downloads/eigen \
                  -I../include/tf/1.11/tensorflow/contrib/makefile/downloads/absl \
                  -I../include/tf/1.11/ \
                  $(shell pkg-config --cflags opencv) 


INCLUDE_LIB    := \
                -L../lib/protobuf/16.0.0  -L../lib/tensorflow/1.11/ubuntu/14.04  \
                 -lprotobuf -ltensorflow_cc -ltensorflow_framework \
                 $(shell pkg-config --libs   opencv) -lpthread

RUNTIME_LINK_LIB    := -Wl,-rpath=../lib/tensorflow/1.11/ubuntu/14.04


# usage :
# make xxx DEFINE_CC_PARAMS="param1 param2 param3 ..."
# it will translate to 
#     -Dparam1  -Dparam2  -Dparam3
#     then put it into compile command line
DEFINE_CC_PARAMS  ?=  

PREFIX_CC_DEFINE_PARAMS  =  $(foreach item, $(DEFINE_CC_PARAMS),  -D$(item) )

CC_OPTIONS = -Wall  -nostartfiles -c -g  -std=c++11  -pthread \
              -Wfatal-errors




%.o: %.cc
	$(CC)  $(INCLUDE_HEADER) $< -o $@  $(CC_OPTIONS) $(PREFIX_CC_DEFINE_PARAMS)

%.out: %.o  dataset_handler.o  models.o 
	$(LD)  $^ -o $@  $(INCLUDE_LIB)  $(RUNTIME_LINK_LIB)




clean:
	@rm -rf  *.o *.out 


help:
	@echo "----- help document -----"
	@echo "make train.out "
	@echo ""
	@echo "make lane_prediction_test.out  "
	@echo ""
	@echo "./train.out  cluster_labels_60_grps   ./  0.00004  0.000001  0.0011  0.0625  0.1245  32 8  800 "
	@echo ""
	@echo "./lane_prediction_test.out  cluster_labels3_30_grps  . "
	@echo ""
	@echo ""


