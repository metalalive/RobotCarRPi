#ifndef __DATASET_HANDLER_H
#define __DATASET_HANDLER_H
#include <stdlib.h> // for exit()
#include <dirent.h> // to check file entries of a folder

#include <cstdlib> // for std::rand , std::srand
#include <ctime>   // for std::time
#include <string>
#include <vector>  // for std::vector
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include "tensorflow/core/framework/tensor.h"

#define ORIGINAL_IMAGE_WIDTH  160
#define ORIGINAL_IMAGE_HEIGHT 60
#define NORMALIZE_NUMBER(num, maxval, minval)  (num*2.0/(maxval-minval))-1

typedef struct point2D {
    float x;
    float y;
} point2D;

typedef struct laneinfo {
    std::string imgpath;
    point2D label ;
    point2D pred  ;
    point2D error  ;
} laneinfo;




class dataset_handler{

    public:

        dataset_handler (std::unordered_map< std::string, int> hyparams_int);

        ~dataset_handler ();

        std::vector< std::vector<laneinfo> > split_shuffle (std::string labels_dir_path="./" );

        void  load_labeled_examples(std::vector<laneinfo> &sliced_samples,
                       tensorflow::Tensor &sample_data, 
                       tensorflow::Tensor &label_data );

        void  load_unlabeled_example ( cv::Mat& img_in,  tensorflow::Tensor &sample_data);

        void  preprocess_one_image (cv::Mat& img_in, cv::Mat& img_out);

    private:
        std::unordered_map< std::string, int>          __hyparams_int;
};


#endif // __DATASET_HANDLER_H
