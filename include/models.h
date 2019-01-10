#ifndef __MODELS_H
#define __MODELS_H

#include <stdlib.h> // for exit()
#include <string>
#include <vector>  // for std::vector
#include <unordered_map>
#include <algorithm>  // for std::min

#include <opencv2/opencv.hpp>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
////#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/public/session_options.h"

#include "tensorflow/core/framework/tensor.h"
#ifndef HOST_OS_RPI
#include "tensorflow/core/util/tensor_slice_writer.h"
#endif // end of n HOST_OS_RPI
#include "tensorflow/core/util/tensor_slice_reader.h"

#include <dataset_handler.h>


// in this neural network model, we only define 2 fully-connected layers
// , currently no convolution layer included (since it won't improve accurancy of predition)
//
// hyoer parameters include :
// --- number of neurons in 1st hidden layer
// --- number of neurons in 2nd hidden layer
// --- maximum value of initial parameters in 1st hidden layer
// --- maximum value of initial parameters in 2nd hidden layer
// --- learning rate
// --- regularization term lambda
// --- number of images in a training/cv batch
class models
{
    private:
        bool __restore_mode;
        std::unordered_map< std::string, std::string>  __hyparams_str;
        std::unordered_map< std::string, float>        __hyparams_flt;
        std::unordered_map< std::string, int>          __hyparams_int;

        float loss_val;

        tensorflow::GraphDef __graph;

        ////std::vector< tensorflow::ClientSession* > sessions;
        std::vector< tensorflow::Session* > sessions;
        std::vector<float> cv_accurancy_all_sessions;

        // our base model has 2 fully-connected layers
        void build ();

        void set_loss (float setval=0.0);

        float get_loss ();

        void calculate_cv_accurancy (std::vector<laneinfo> samples, float cv_threshold=0.20) ;


    public:

        models( bool restore_mode,
                std::unordered_map< std::string, std::string> hyparams_str, 
                std::unordered_map< std::string, float>       hyparams_flt, 
                std::unordered_map< std::string, int>         hyparams_int ) ;

        ~models ();

        // start a new session with respect to tensorflow::Scope,
        void initialize();

        int save_to_file();

        int restore_from_file();

        float  train (std::vector<laneinfo> &samples, dataset_handler &dl);

        float  validate (std::vector<laneinfo> &samples, dataset_handler &dl);

        void   predict (cv::Mat &img_in, point2D &out, dataset_handler &dl);

        float  get_cv_accurancy ();

};


#endif // __MODELS_H
