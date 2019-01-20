#include <models.h>

models::models( bool restore_mode ,
                std::unordered_map< std::string, std::string> hyparams_str, 
                std::unordered_map< std::string, float>       hyparams_flt, 
                std::unordered_map< std::string, int>         hyparams_int ) :
                __restore_mode (restore_mode),
                __hyparams_str (hyparams_str),
                __hyparams_flt (hyparams_flt),
                __hyparams_int (hyparams_int)
{

    if (__restore_mode) {
        restore_from_file( );
    }
    else {
        build() ;
    }
}


models::~models()
{
    for (tensorflow::Session *sess : sessions) {
        delete sess;
    }
    sessions.clear();
}


void models::build ()
{
#ifndef HOST_OS_RPI
    int num_fc_inl = __hyparams_int["num_fc_inl"];
    int num_fc_l1  = __hyparams_int["num_fc_l1"];
    int num_fc_l2  = __hyparams_int["num_fc_l2"];
    int num_fc_outl= __hyparams_int["num_fc_outl"];
    float max_val_param_l1  = __hyparams_flt["max_val_param_l1"] ;
    float max_val_param_l2  = __hyparams_flt["max_val_param_l2"] ;
    float max_val_param_out = __hyparams_flt["max_val_param_out"] ;

    tensorflow::Scope  root = tensorflow::Scope::NewRootScope();

    tensorflow::Output sample_input_entry = tensorflow::ops::Placeholder (root.WithOpName("sample_entry"), tensorflow::DataTypeToEnum<float>::v() );
    tensorflow::Output label_input_entry  = tensorflow::ops::Placeholder (root.WithOpName("label_entry"),  tensorflow::DataTypeToEnum<float>::v() );

    // define parameter matrices
    tensorflow::Output w1 = tensorflow::ops::Variable (root.WithOpName("w1"), {num_fc_inl, num_fc_l1}, tensorflow::DT_FLOAT);
    tensorflow::Output w2 = tensorflow::ops::Variable (root.WithOpName("w2"), {num_fc_l1 , num_fc_l2}, tensorflow::DT_FLOAT);
    tensorflow::Output w3 = tensorflow::ops::Variable (root.WithOpName("w3"), {num_fc_l2 , num_fc_outl}, tensorflow::DT_FLOAT);
    tensorflow::Output b1 = tensorflow::ops::Variable (root.WithOpName("b1"), {1, num_fc_l1},  tensorflow::DT_FLOAT);
    tensorflow::Output b2 = tensorflow::ops::Variable (root.WithOpName("b2"), {1, num_fc_l2},  tensorflow::DT_FLOAT);
    tensorflow::Output b3 = tensorflow::ops::Variable (root.WithOpName("b3"), {1, num_fc_outl},tensorflow::DT_FLOAT);

    // assign operation, for assigning initial random values to layers
    // tensorflow::ops::RandomXXX cannot accept unsigned integers in its shape arguments
    // the shape has to be integers . e.g. int num_fc_inl, NOT unsigned num_fc_inl
    // Doing it wrong will cause the subsequent operations added to the same scope bail out.
    tensorflow::Output w1_rand_0to1 = tensorflow::ops::RandomUniform(root, {num_fc_inl, num_fc_l1},  tensorflow::DT_FLOAT);
    tensorflow::Output w2_rand_0to1 = tensorflow::ops::RandomUniform(root, {num_fc_l1, num_fc_l2},   tensorflow::DT_FLOAT);
    tensorflow::Output w3_rand_0to1 = tensorflow::ops::RandomUniform(root, {num_fc_l2, num_fc_outl}, tensorflow::DT_FLOAT);
    tensorflow::Output b1_rand_0to1 = tensorflow::ops::RandomUniform(root, {1, num_fc_l1},           tensorflow::DT_FLOAT);
    tensorflow::Output b2_rand_0to1 = tensorflow::ops::RandomUniform(root, {1, num_fc_l2},           tensorflow::DT_FLOAT);
    tensorflow::Output b3_rand_0to1 = tensorflow::ops::RandomUniform(root, {1, num_fc_outl},         tensorflow::DT_FLOAT);

    // Be sure to randomly initialize the parameters for the first time to train the model.
    tensorflow::Output maxval_ops_l1  = tensorflow::ops::Cast(root, max_val_param_l1 , tensorflow::DT_FLOAT);
    tensorflow::Output maxval_ops_l2  = tensorflow::ops::Cast(root, max_val_param_l2 , tensorflow::DT_FLOAT);
    tensorflow::Output maxval_ops_out = tensorflow::ops::Cast(root, max_val_param_out, tensorflow::DT_FLOAT);
    
    tensorflow::Output w1_rand_init = tensorflow::ops::Multiply(root, w1_rand_0to1, maxval_ops_l1 );
    tensorflow::Output w2_rand_init = tensorflow::ops::Multiply(root, w2_rand_0to1, maxval_ops_l2 );
    tensorflow::Output w3_rand_init = tensorflow::ops::Multiply(root, w3_rand_0to1, maxval_ops_out);
    tensorflow::Output b1_rand_init = tensorflow::ops::Multiply(root, b1_rand_0to1, maxval_ops_l1 );
    tensorflow::Output b2_rand_init = tensorflow::ops::Multiply(root, b2_rand_0to1, maxval_ops_l2 );
    tensorflow::Output b3_rand_init = tensorflow::ops::Multiply(root, b3_rand_0to1, maxval_ops_out);
    tensorflow::Output assigned_rand_w1 = tensorflow::ops::Assign (root.WithOpName("assigned_rand_w1"), w1, w1_rand_init);
    tensorflow::Output assigned_rand_w2 = tensorflow::ops::Assign (root.WithOpName("assigned_rand_w2"), w2, w2_rand_init);
    tensorflow::Output assigned_rand_w3 = tensorflow::ops::Assign (root.WithOpName("assigned_rand_w3"), w3, w3_rand_init);
    tensorflow::Output assigned_rand_b1 = tensorflow::ops::Assign (root.WithOpName("assigned_rand_b1"), b1, b1_rand_init);
    tensorflow::Output assigned_rand_b2 = tensorflow::ops::Assign (root.WithOpName("assigned_rand_b2"), b2, b2_rand_init);
    tensorflow::Output assigned_rand_b3 = tensorflow::ops::Assign (root.WithOpName("assigned_rand_b3"), b3, b3_rand_init);
    // assign operation, for specifying trained parameters to layers
    auto trained_w1 = tensorflow::ops::Placeholder (root.WithOpName("chkptr0_entry"), tensorflow::DataTypeToEnum<float>::v()); 
    auto trained_w2 = tensorflow::ops::Placeholder (root.WithOpName("chkptr1_entry"), tensorflow::DataTypeToEnum<float>::v()); 
    auto trained_w3 = tensorflow::ops::Placeholder (root.WithOpName("chkptr2_entry"), tensorflow::DataTypeToEnum<float>::v()); 
    auto trained_b1 = tensorflow::ops::Placeholder (root.WithOpName("chkptr3_entry"), tensorflow::DataTypeToEnum<float>::v()); 
    auto trained_b2 = tensorflow::ops::Placeholder (root.WithOpName("chkptr4_entry"), tensorflow::DataTypeToEnum<float>::v()); 
    auto trained_b3 = tensorflow::ops::Placeholder (root.WithOpName("chkptr5_entry"), tensorflow::DataTypeToEnum<float>::v()); 
    auto assigned_trained_w1 = tensorflow::ops::Assign (root.WithOpName("assigned_chkptr0"), w1, trained_w1);
    auto assigned_trained_w2 = tensorflow::ops::Assign (root.WithOpName("assigned_chkptr1"), w2, trained_w2);
    auto assigned_trained_w3 = tensorflow::ops::Assign (root.WithOpName("assigned_chkptr2"), w3, trained_w3);
    auto assigned_trained_b1 = tensorflow::ops::Assign (root.WithOpName("assigned_chkptr3"), b1, trained_b1);
    auto assigned_trained_b2 = tensorflow::ops::Assign (root.WithOpName("assigned_chkptr4"), b2, trained_b2);
    auto assigned_trained_b3 = tensorflow::ops::Assign (root.WithOpName("assigned_chkptr5"), b3, trained_b3);

    tensorflow::Output z1 = tensorflow::ops::Add(
                                         root.WithOpName("z1"), 
                                         tensorflow::ops::MatMul(root, sample_input_entry, w1),
                                         b1 );
    tensorflow::Output a1 = tensorflow::ops::Tanh(root.WithOpName("a1"), z1);
    tensorflow::Output z2 = tensorflow::ops::Add(
                                         root.WithOpName("z2"), 
                                         tensorflow::ops::MatMul(root, a1, w2), b2 );
    tensorflow::Output a2 = tensorflow::ops::Tanh(root.WithOpName("a2"), z2);
    tensorflow::Output outlayer = tensorflow::ops::Add(
                                         root.WithOpName("outlayer"), 
                                         tensorflow::ops::MatMul(root, a2, w3), b3 );
     

    // define cost function : assume there are k neurons in output layer
    //                        say y1, y2, y3, ...... yk
    // 
    //     square_error = 
    //           (y1_true - y1_pred) ^ 2 
    //           (y2_true - y2_pred) ^ 2 
    //           (y3_true - y3_pred) ^ 2 
    //           .....
    //
    //     sum_of_square_error =
    //           (y1_true - y1_pred) ^ 2 + (y2_true - y2_pred) ^ 2 + .....
    //
    //     l2_regulariztion =
    //            1/2 * sum(w1 ^ 2 + w2 ^ 2 + w3 ^ 2 + ....)
    //
    //
    //     cost function = sum_of_square_error + lambda * l2_regulariztion
    //
    // here we don't estimate mean of sum_of_square_error using tensorflow framework like tensorflow::ops::Mean ,
    // instead we will do it after we get sum_of_square_error for entire training dataset,
    // since we'll apply stochastic gradient descent to the training procedure.
    // feed training examples one by one , to update the gradient.

    tensorflow::Output reg_lambda = tensorflow::ops::Cast(
                                        root, __hyparams_flt["reg_lambda"],
                                        tensorflow::DT_FLOAT);

    tensorflow::Output pred_errors = tensorflow::ops::Sub (root.WithOpName("pred_errors"), outlayer, label_input_entry );

    tensorflow::Output  sum_square_error = tensorflow::ops::Sum (
                                               root, tensorflow::ops::Square(root, pred_errors),  {0,1}
                                          );

    tensorflow::Output regularization = tensorflow::ops::Mul(
                                    root, reg_lambda,
                                       tensorflow::ops::AddN (
                                              root.WithOpName("regularization"), 
                                              // L2Loss does the following equation :
                                              // 1/2 * sum( w ^ 2 )
                                              std::initializer_list<tensorflow::Input>{
                                                  tensorflow::ops::L2Loss (root, w1),
                                                  tensorflow::ops::L2Loss (root, w2),
                                                  tensorflow::ops::L2Loss (root, w3)
                                              }
                                          )
                                );

    tensorflow::Output cost_fn = tensorflow::ops::Add (root.WithOpName("cost_fn"), sum_square_error,  regularization );

    // ----- for gradient descent -----
    // add gradients of the forward propagation, from the cost function to the graph, 
    // with regards to each parameters
    // AddSymbolicGradients will initialize grad_out , 
    // grad_out will be filled with nodes, which give the gradient for a tensorflow::Variable
    tensorflow::Output learning_rate = tensorflow::ops::Cast(
                                        root, __hyparams_flt["learning_rate"],
                                        tensorflow::DT_FLOAT);
 
    // https://github.com/tensorflow/tensorflow/issues/18149
    // the tensorflow issue above helps to dump more useful message to check if your graph is ok for subsequent execution
    if (! root.ok()) {
        std::cout << "[ERROR] ----- Problems are found in the model ----- "<< std::endl;
        LOG(FATAL) << root.status().ToString();
        exit(EXIT_FAILURE);
    } 

    std::vector<tensorflow::Output> grad_out;

    TF_CHECK_OK(
        tensorflow::AddSymbolicGradients(root, {cost_fn}, {w1, b1, w2, b2, w3, b3}, &grad_out )
    );
    // give paritial derivative of the cost function with respect to each parameter matrix (tensorflow::Variable)
    tensorflow::Output grad_w1 = tensorflow::ops::ApplyGradientDescent( root.WithOpName("grad_w1"), w1, learning_rate, {grad_out[0]} );    
    tensorflow::Output grad_b1 = tensorflow::ops::ApplyGradientDescent( root.WithOpName("grad_b1"), b1, learning_rate, {grad_out[1]} );    
    tensorflow::Output grad_w2 = tensorflow::ops::ApplyGradientDescent( root.WithOpName("grad_w2"), w2, learning_rate, {grad_out[2]} );    
    tensorflow::Output grad_b2 = tensorflow::ops::ApplyGradientDescent( root.WithOpName("grad_b2"), b2, learning_rate, {grad_out[3]} );    
    tensorflow::Output grad_w3 = tensorflow::ops::ApplyGradientDescent( root.WithOpName("grad_w3"), w3, learning_rate, {grad_out[4]} );    
    tensorflow::Output grad_b3 = tensorflow::ops::ApplyGradientDescent( root.WithOpName("grad_b3"), b3, learning_rate, {grad_out[5]} );    

    TF_CHECK_OK(root.ToGraphDef(&__graph));
#endif // end of n HOST_OS_RPI
}



void models::initialize()
{
    ////tensorflow::ClientSession *sess = new tensorflow::ClientSession (root);
    tensorflow::SessionOptions options;
    tensorflow::Session *sess;
    sess = tensorflow::NewSession(options);
    // create new session from graphdef
    if (sess != nullptr) {
        sess->Create(__graph);
        sessions.push_back (sess);
    }
    else {
        std::cout << "[ERROR] failed to create session in restore mode." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (__restore_mode) {
        // load trained parameters (checkpoint)
        unsigned int idx = 0;
        std::vector<std::string> tensor_names = {"chkptr0", "chkptr1", "chkptr2", "chkptr3", "chkptr4", "chkptr5"} ;
        std::string entry_name      ;
        std::string target_ops_name ;
        tensorflow::TensorShape shape;
        tensorflow::DataType    type;
        std::unique_ptr<tensorflow::Tensor> out_t ;

        tensorflow::checkpoint::TensorSliceReader reader (__hyparams_str["trained_param_path"]) ;
        TF_CHECK_OK( reader.status() );
        for (idx=0; idx<tensor_names.size() ; idx++)
        {
            if(reader.HasTensor(tensor_names.at(idx), &shape, &type))
            {
                reader.GetTensor( tensor_names.at(idx), &out_t );
                std::string entry_name = tensor_names.at(idx) + "_entry";
                std::string target_ops_name = "assigned_" + tensor_names.at(idx);
                TF_CHECK_OK(
                    sessions.back()->Run({{entry_name, *out_t}}, {target_ops_name}, {}, nullptr)
                );
            }
        }
    }
    else {
        TF_CHECK_OK(
            sessions.back()->Run({}, {
                                       "assigned_rand_w1", "assigned_rand_b1",
                                       "assigned_rand_w2", "assigned_rand_b2",
                                       "assigned_rand_w3", "assigned_rand_b3"
                                     }, {}, nullptr)
        );
    }
}



float  models::train (std::vector<laneinfo>  &samples, dataset_handler &dl)
{
#ifdef HOST_OS_RPI
    return 0.f;
#else
    int num_examples = samples.size();
    int num_examples_in_curr_batch = 0;
    int num_samples_batch = __hyparams_int["num_samples_batch"];
    int num_fc_inl  = __hyparams_int["num_fc_inl"];
    int num_fc_outl = __hyparams_int["num_fc_outl"];
    int samples_start_idx = 0;
    int samples_end_idx = 0;
    float curr_loss = 0;
    float accu_loss = 0;

    std::vector<laneinfo>::iterator  start_sample;
    std::vector<laneinfo>::iterator  end_sample;
    std::vector<laneinfo>            sliced_samples;
    std::vector<tensorflow::Tensor>  eval_result;
    
    set_loss(0.0);
    if (num_examples > 0 )
    {
        for (samples_start_idx=0; samples_start_idx<num_examples; samples_start_idx+=num_samples_batch) 
        {
            samples_end_idx = std::min( samples_start_idx + num_samples_batch , num_examples );
            num_examples_in_curr_batch = samples_end_idx - samples_start_idx ;
            start_sample = samples.begin() + samples_start_idx;
            end_sample   = samples.begin() + samples_end_idx ;
            sliced_samples = std::vector<laneinfo> (start_sample, end_sample);
            tensorflow::TensorShape sample_data_shape {num_examples_in_curr_batch, num_fc_inl};
            tensorflow::TensorShape  label_data_shape {num_examples_in_curr_batch, num_fc_outl};
            tensorflow::Tensor sample_data (tensorflow::DataTypeToEnum<float>::v(), sample_data_shape);
            tensorflow::Tensor  label_data (tensorflow::DataTypeToEnum<float>::v(), label_data_shape);
            // copy image data from cv::Mat to tensorflow::Tensor
            dl.load_labeled_examples(sliced_samples, sample_data, label_data);
            // check first few items to see if pixels are copied successfully.
            eval_result.clear();
            // get loss of each batch
            TF_CHECK_OK(
                sessions.back()->Run( 
                    {{"sample_entry", sample_data}, {"label_entry", label_data}},
                    {"cost_fn"}, {}, &eval_result ) 
            );

            // accumulate each loss
            curr_loss = *(float*) eval_result[0].scalar<float>().data();
            accu_loss = get_loss() + curr_loss;
            set_loss( accu_loss );
            TF_CHECK_OK(
                sessions.back()->Run(
                    {{"sample_entry",sample_data}, {"label_entry", label_data}},
                    {"grad_w1", "grad_w2", "grad_w3", "grad_b1", "grad_b2", "grad_b3"},
                    {}, nullptr
                )
            );
        }
    }
    float final_loss = (get_loss() * 1.0) / num_examples;
    return final_loss ;
#endif // end of HOST_OS_RPI
}



// In validate() , we need to extract the value of each item of the matrix,
// which is tensorflow::Tensor, where the calculation really happened on the graph
// tensorflow::Tensor applies Eigen::Tensor so users can read the value of each item of a matrix
// in Matlab-like coding style.
// Here is an example to do so:
//
// tensorflow::Tensor eval_result; // assume it's estimated by session.Run()
//
// for (int jdx = 1 ; jdx < eval_result.size(); jdx++)
// {
//     Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex> layerval = eval_result[jdx].matrix<float>() ;
//     std::cout << "[DBG][train] check : eval_result["<< jdx <<"]"
//               << eval_result[jdx].dim_size(0) <<"x"
//               << eval_result[jdx].dim_size(1) <<"\t:"
//               << layerval(2,0) <<", "<< layerval(2,0) <<", "
//               << layerval(7,1) <<", "<< layerval(7,1) << std::endl;
// }
// LOG(INFO) << eval_result[0].DebugString();

float  models::validate (std::vector<laneinfo> &samples,  dataset_handler &dl)
{
    int num_examples = samples.size();
    int num_examples_in_curr_batch = 0;
    int num_samples_batch = __hyparams_int["num_samples_batch"];
    int num_fc_inl  = __hyparams_int["num_fc_inl"];
    int num_fc_outl = __hyparams_int["num_fc_outl"];
    int samples_start_idx = 0;
    int samples_end_idx = 0;
    float curr_loss = 0;
    float accu_loss = 0;

    std::vector<laneinfo>::iterator  start_sample;
    std::vector<laneinfo>::iterator  end_sample;
    std::vector<laneinfo>            sliced_samples;
    std::vector< tensorflow::Tensor > eval_result;

    int idx = 0;
    
    set_loss(0.0);
    if (num_examples > 0)
    {
        for (samples_start_idx=0; samples_start_idx<num_examples; samples_start_idx+=num_samples_batch) 
        {
            samples_end_idx = std::min( samples_start_idx + num_samples_batch , num_examples );
            num_examples_in_curr_batch = samples_end_idx - samples_start_idx ;
            start_sample = samples.begin() + samples_start_idx;
            end_sample   = samples.begin() + samples_end_idx ;
            sliced_samples = std::vector<laneinfo> (start_sample, end_sample);
            tensorflow::TensorShape sample_data_shape {num_examples_in_curr_batch, num_fc_inl};
            tensorflow::TensorShape  label_data_shape {num_examples_in_curr_batch, num_fc_outl};
            tensorflow::Tensor sample_data (tensorflow::DataTypeToEnum<float>::v(), sample_data_shape);
            tensorflow::Tensor  label_data (tensorflow::DataTypeToEnum<float>::v(), label_data_shape);
            dl.load_labeled_examples(sliced_samples, sample_data, label_data);
            eval_result.clear();
            // get loss of each batch
            TF_CHECK_OK(
                sessions.back()->Run( 
                    {{"sample_entry", sample_data}, {"label_entry", label_data}}, 
                    {"cost_fn", "outlayer"}, {}, &eval_result 
                )
            );
            // accumulate each loss
            curr_loss = *(float*) eval_result[0].scalar<float>().data();
            accu_loss = get_loss() + curr_loss;
            set_loss( accu_loss );
            // collecting predictions 
            Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex> pred_mtx = eval_result[1].matrix<float>() ;
            for (idx=0; idx < eval_result[1].dim_size(0); idx++ )
            {
                samples.at(samples_start_idx + idx).pred.x   = std::floor(1000 * pred_mtx(idx,0)) / 1000;
                samples.at(samples_start_idx + idx).pred.y   = std::floor(1000 * pred_mtx(idx,1)) / 1000;
                samples.at(samples_start_idx + idx).error.x = std::fabs(samples.at(samples_start_idx + idx).pred.x - samples.at(samples_start_idx + idx).label.x);
                samples.at(samples_start_idx + idx).error.y = std::fabs(samples.at(samples_start_idx + idx).pred.y - samples.at(samples_start_idx + idx).label.y);
            }
        }
    }
    calculate_cv_accurancy (samples, 0.25); 

    float final_loss = (get_loss() * 1.0) / num_examples;
    return final_loss;
}




void  models::predict (cv::Mat &img_in, point2D &out, dataset_handler &dl)
{
    int num_fc_inl  = __hyparams_int["num_fc_inl"];
    std::vector< tensorflow::Tensor > eval_result;
    tensorflow::TensorShape sample_data_shape {1, num_fc_inl};
    tensorflow::Tensor sample_data (tensorflow::DataTypeToEnum<float>::v(), sample_data_shape);
    // pre-process image then copy to Tensor.
    dl.load_unlabeled_example ( img_in,  sample_data );
    TF_CHECK_OK(
        sessions.back()->Run ( {{"sample_entry", sample_data}},  {"outlayer"}, {}, &eval_result )
    );
    // get predicted value
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex> pred_mtx = eval_result[0].matrix<float>();
    out.x = pred_mtx(0,0);
    out.y = pred_mtx(0,1);
}




void models::calculate_cv_accurancy (std::vector<laneinfo> samples, float cv_threshold ) 
{
    float accurancy = 0.0;
    float x_threshold = cv_threshold * 1.0;
    float y_threshold = cv_threshold * 1.5;
    int num_passed   = 0;
    int num_examples = samples.size();
    int idx = 0;

    for (idx=0; idx<num_examples; idx++)
    {
        if (samples.at(idx).error.x < x_threshold && samples.at(idx).error.y < y_threshold) {
            num_passed ++;
        }
    }
    accurancy = (1.0 * num_passed) / num_examples;
    if (sessions.size() > cv_accurancy_all_sessions.size()) 
    {
        cv_accurancy_all_sessions.push_back (accurancy);
    }
    else if (sessions.size() == cv_accurancy_all_sessions.size()) 
    {
        cv_accurancy_all_sessions.back () = accurancy;
    }
}


float models::get_cv_accurancy ()
{
    return  cv_accurancy_all_sessions.back() ;
}



void models::set_loss (float setval) 
{
    loss_val = setval;
}

float models::get_loss () 
{
    return loss_val;
}


// name mapping between vector of Tensors and checkpoint name in chackpoint file...
//     w1 --> chkptr0
//     w2 --> chkptr1
//     w3 --> chkptr2 
//     b1 --> chkptr3 
//     b2 --> chkptr4 
//     b3 --> chkptr5 
//
int models::save_to_file()
{
#ifndef HOST_OS_RPI
    std::vector<tensorflow::Tensor>  trained_param_vals;
    float min_accurancy = 9999.99;
    unsigned int   min_accurancy_idx = 0;
    unsigned int   idx = 0;
    unsigned int   jdx = 0;

    for ( idx=0; idx < cv_accurancy_all_sessions.size(); idx++)
    {
        if (min_accurancy > cv_accurancy_all_sessions.at(idx)) 
        {
            min_accurancy = cv_accurancy_all_sessions.at(idx);
            min_accurancy_idx = idx;
        }
    }
    if (sessions.at(min_accurancy_idx) != nullptr)
    {
        // get current values of trained oarameters.
        TF_CHECK_OK(
            sessions.at(min_accurancy_idx)->Run( 
                {}, {"w1", "w2", "w3", "b1", "b2", "b3"}, {}, &trained_param_vals) 
        );
        // write trained parameters to file
        tensorflow::checkpoint::TensorSliceWriter writer (
            __hyparams_str["trained_param_path"],
            tensorflow::checkpoint::CreateTableTensorSliceBuilder	
        );
        jdx = 0;
        std::string chkptrname ;
        tensorflow::TensorSlice tslice = tensorflow::TensorSlice::ParseOrDie("-:-");
        const float* train_param_rawdata = nullptr;
        for (tensorflow::Tensor t : trained_param_vals) {
            train_param_rawdata = reinterpret_cast<const float*>(t.tensor_data().data());
            tensorflow::TensorShape shape({t.dim_size(0), t.dim_size(1)});
            chkptrname = std::string("chkptr") + std::to_string(jdx);
            TF_CHECK_OK(
                writer.Add<float>(chkptrname, shape, tslice, train_param_rawdata)
            );
            jdx ++;
        }
        TF_CHECK_OK(writer.Finish());
        // write neural network model
        TF_CHECK_OK(
            tensorflow::WriteTextProto(
                tensorflow::Env::Default() , __hyparams_str["model_path"],   __graph
            )
        );
    }
    else
    {
        std::cout << "[ERROR] session #"<< min_accurancy_idx <<" is NOT available." << std::endl;
        return 1;
    }
#endif // end of n HOST_OS_RPI
    return 0;
}



int models::restore_from_file()
{
    TF_CHECK_OK(
        tensorflow::ReadTextProto(tensorflow::Env::Default(), __hyparams_str["model_path"], &__graph)
    );
    return 0;
}


