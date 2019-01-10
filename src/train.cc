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
    printf("    trainer  <EXAMPLE_PATH>   <OUTPUT_TRAINED_MODEL_PATH>   \r\n");
    printf("             <LEARNING_RATE>  <LAMBDA> \r\n");
    printf("             <MAX_VAL_PARAM_L1>  <MAX_VAL_PARAM_L2> <MAX_VAL_PARAM_OUT> \r\n");
    printf("             <NUM_NEURONS_L1>  <NUM_NEURONS_L2>  <NUM_TRAIN_ITERATION>  \r\n");
    printf("\r\n");
    printf("\r\n");
    printf(" essential arguments:                               \r\n"); 
    printf("\r\n"); 
    printf("    EXAMPLE_PATH,                                   \r\n"); 
    printf("                 the path containing all examples,  \r\n"); 
    printf("                 each examples specifies the location of its image  \r\n"); 
    printf("               , and a point (x,y) on image indicating the centroid of the lane. \r\n"); 
    printf("    OUTPUT_TRAINED_MODEL_PATH                                          \r\n"); 
    printf("               the location to save the trained neural network model.  \r\n"); 
    printf("               in Tensorflow C++ API, this includes 2 parts :          \r\n"); 
    printf("                   * neural network model (protobuf)                   \r\n"); 
    printf("                   * trained parameter matrices (checkpoints)          \r\n"); 
    printf("    MAX_VAL_PARAM_L1                                                   \r\n"); 
    printf("               range of initial random value in hidden layer 1 \r\n"); 
    printf("    MAX_VAL_PARAM_L2                                                   \r\n"); 
    printf("               range of initial random value in hidden layer 2 \r\n"); 
    printf("    MAX_VAL_PARAM_OUT                                                  \r\n"); 
    printf("               range of initial random value in output layer   \r\n"); 
    printf("\r\n"); 
    printf("    NUM_NEURONS_L1                 \r\n"); 
    printf("               number of neurons in hidden layer 1             \r\n"); 
    printf("    NUM_NEURONS_L2                 \r\n"); 
    printf("               number of neurons in hidden layer 2             \r\n"); 
    printf("    NUM_TRAIN_ITERATION            \r\n"); 
    printf("               number of iteration in training procedure       \r\n"); 
    printf("\r\n"); 
    printf("\r\n"); 
    printf("\r\n"); 
}



int  main (int argc, char ** argv) 
{
    if (argc < 10)  {
        help_doc (); return 1;
    }
    std::string labels_path  = std::string (argv[1]);
    std::string output_trained_model_path = std::string (argv[2]);
    output_trained_model_path = output_trained_model_path + "/";

    unsigned short int  resized_img_width = 40;
    unsigned short int  resized_img_height = 15;
    unsigned short int  img_chnal = 3;
    unsigned short int  round = 2;
    unsigned short int  num_train_iteration = std::stoi(argv[10]); // 800~ 
    float train_loss = 0.f;
    float cv_loss    = 0.f;
    float accurancy  = 0.f;

    unsigned short int idx = 0;
    unsigned short int jdx = 0;

    std::unordered_map< std::string, std::string > hyparams_str ;
    std::unordered_map< std::string, float > hyparams_flt ;
    std::unordered_map< std::string, int >   hyparams_int ;

    hyparams_flt["learning_rate"]  = std::stof(argv[3]); // 0.00004
    hyparams_flt["reg_lambda"]     = std::stof(argv[4]); // 0.000001
    hyparams_flt["max_val_param_l1"]  = std::stof(argv[5]); // 0.0011
    hyparams_flt["max_val_param_l2"]  = std::stof(argv[6]); // 0.0625
    hyparams_flt["max_val_param_out"] = std::stof(argv[7]); // 0.124
    hyparams_int["resized_img_width"]  = resized_img_width;
    hyparams_int["resized_img_height"] = resized_img_height;
    hyparams_int["num_fc_inl"] = resized_img_width * resized_img_height * img_chnal;
    hyparams_int["num_fc_l1"] = std::stoi(argv[8]); // 32
    hyparams_int["num_fc_l2"] = std::stoi(argv[9]); // 8
    hyparams_int["num_fc_outl"] = 2;
    hyparams_int["num_samples_batch"] = 300;
    hyparams_str["model_path"]         = output_trained_model_path + "models.pbtxt";
    hyparams_str["trained_param_path"] = output_trained_model_path + "trained_param_checkpoints.bin";

    dataset_handler dl (hyparams_int);
    models nnm {false, hyparams_str, hyparams_flt, hyparams_int};
    std::vector< std::vector<laneinfo> > samples ;

    for (idx=0; idx<round; idx++) {
        samples = dl.split_shuffle (labels_path);
        nnm.initialize();

        std::cout << "[INFO] round "<< idx <<", # train/cv/test examples in dataset = "
                  << samples.at(0).size() << "/"<< samples.at(1).size() << "/"
                  << samples.at(2).size() << std::endl;

        for (jdx=0; jdx<num_train_iteration; jdx++) {
            train_loss = nnm.train (samples.at(0), dl);
            if (jdx % 10 == 0) {
                cv_loss    = nnm.validate (samples.at(1), dl);
                std::cout << "     iter. "<< jdx <<"\t, train loss: "<< train_loss << "\t";
                std::cout << "     cv loss: "<< cv_loss << std::endl;
            }
        }
        accurancy = nnm.get_cv_accurancy ();
        std::cout << "[INFO] accurancy : "<< (accurancy*100) << "%" << std::endl ;
        std::cout << "[INFO] train loss: "<< train_loss << "\t" << "cv loss: "<< cv_loss << std::endl;
        std::sort( samples.at(1).begin(), samples.at(1).end(), sort_by_error() );
        report_prediction (samples.at(1));

        std::cout << std::endl;
        
        for (jdx=0; jdx<samples.size(); jdx++) {
            samples.at(jdx).clear();
        }
        samples.clear();
    }

    nnm.save_to_file();

}



