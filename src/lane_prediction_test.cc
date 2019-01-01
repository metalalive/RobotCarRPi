// this applicaiton is supposed to run on raspberry pi
//
// what it will do :
//     #1 load the model we previously defined, 
//        and parameter matrices we previously trained in models.
//     #2 predict lane curvature by checking input image one by one
//     #3 estimate avg. execution time of each prediction
//     #4 estimate accurancy, sort the validation examples by their errors

#include <iostream>
#include <sstream> // for std::streamtring
#include <vector>  // for std::vector
#include <cstdlib> // for std::fabs
#include <string> 
#include <unordered_map>

#include <dataset_handler.h>
#include <models.h>


struct sort_by_error {
    bool operator () (const laneinfo &a, const laneinfo &b) {
        float a_err = 0;
        float b_err = 0;
        a_err = a.error.x + a.error.y ;
        b_err = b.error.x + b.error.y ;
        return a_err > b_err;
    }
};


void report_prediction (std::vector<laneinfo>& samples )
{
    std::cout << "[INFO] list first 30 predictions with errors :" << std::endl;
    std::cout << "label (x,y) \t\t predict (x,y) \t\t error(x,y) \t\t image path" << std::endl;
    for (int pdx=0; pdx<30; pdx++)
    {
        std::cout << "|"<< samples.at(pdx).label.x <<"\t|"<< samples.at(pdx).label.y;
        std::cout << "\t|"<< samples.at(pdx).pred.x  <<"\t|"<< samples.at(pdx).pred.y ;
        std::cout << "\t|"<< samples.at(pdx).error.x <<"\t|"<< samples.at(pdx).error.y ;
        std::cout << "\t|"<< samples.at(pdx).imgpath << std::endl ;
    }
}



void help_doc () {
    printf("[ERROR] arguments missing \r\n");
    printf("          \r\n");
    printf("load samples and corresponding images to train our neural network ...  \r\n");
    printf("          \r\n");
    printf("[example] \r\n");
    printf("    trainer  <EXAMPLE_PATH>   <INPUT_TRAINED_MODEL_PATH>   \r\n");
    printf("\r\n");
    printf("\r\n");
    printf(" essential arguments:                               \r\n"); 
    printf("\r\n"); 
    printf("    EXAMPLE_PATH,                                   \r\n"); 
    printf("                 the path containing all examples,  \r\n"); 
    printf("                 each examples specifies the location of its image  \r\n"); 
    printf("               , and a point (x,y) on image indicating the centroid of the lane. \r\n"); 
    printf("    INPUT_TRAINED_MODEL_PATH                                           \r\n"); 
    printf("               the location to load the trained neural network model.  \r\n"); 
    printf("               in Tensorflow C++ API, this includes 2 parts :          \r\n"); 
    printf("                   * neural network model (protobuf)                   \r\n"); 
    printf("                   * trained parameter matrices (checkpoints)          \r\n"); 
    printf("\r\n"); 
    printf("\r\n"); 
    printf("\r\n"); 
    printf("\r\n"); 
}


int  main (int argc, char ** argv) 
{
    if (argc < 2)  {
        help_doc (); return 1;
    }
    std::string labels_path  = std::string (argv[1]);
    std::string input_trained_model_path = std::string (argv[2]);
    input_trained_model_path = input_trained_model_path + "/";

    unsigned short int  img_width = 40;
    unsigned short int  img_height = 15;
    unsigned short int  img_chnal = 3;
    float cv_loss    = 0.f;
    float accurancy  = 0.f;

    unsigned short int jdx = 0;

    std::unordered_map< std::string, std::string > hyparams_str ;
    std::unordered_map< std::string, float > hyparams_flt ;
    std::unordered_map< std::string, int >   hyparams_int ;

    hyparams_flt["learning_rate"]  = 0; 
    hyparams_flt["reg_lambda"]     = 0; 
    hyparams_flt["max_val_param_l1"]  = 0 ; 
    hyparams_flt["max_val_param_l2"]  = 0 ; 
    hyparams_flt["max_val_param_out"] = 0 ; 
    hyparams_int["img_width"]  = img_width;
    hyparams_int["img_height"] = img_height;
    hyparams_int["num_fc_inl"] = img_width * img_height * img_chnal;
    hyparams_int["num_fc_l1"] = 0; 
    hyparams_int["num_fc_l2"] = 0; 
    hyparams_int["num_fc_outl"] = 2;
    hyparams_int["num_samples_batch"] = 300;
    hyparams_str["model_path"]         = input_trained_model_path + "models.pbtxt";
    hyparams_str["trained_param_path"] = input_trained_model_path + "trained_param_checkpoints.bin";

    dataset_handler dl (hyparams_int);
    models nnm {true, hyparams_str, hyparams_flt, hyparams_int};
    std::vector< std::vector<laneinfo> > samples ;

    samples = dl.split_shuffle (labels_path);
    nnm.initialize();
    cv_loss    = nnm.validate (samples.at(1), dl);
    accurancy = nnm.get_cv_accurancy ();
    std::sort( samples.at(1).begin(), samples.at(1).end(), sort_by_error() );

    std::cout << "[INFO] accurancy : "<< (accurancy*100) << "%" << std::endl ;
    std::cout << "[INFO] cv loss: "<< cv_loss << std::endl;
    report_prediction (samples.at(1));
    std::cout << std::endl;

    for (jdx=0; jdx<samples.size(); jdx++) {
        samples.at(jdx).clear();
    }
    samples.clear();
    return 0;
}


