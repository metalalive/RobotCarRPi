#include <dataset_handler.h>

dataset_handler::dataset_handler (std::unordered_map< std::string, int> hyparams_int) : __hyparams_int (hyparams_int)
{
}


dataset_handler::~dataset_handler ()
{
}


std::vector< std::vector<laneinfo> > dataset_handler::split_shuffle (std::string labels_dir_path)
{
    // get all the files(labels) in dataset_home
    std::vector< std::vector<laneinfo> >  output;
    std::vector<laneinfo> labels_of_curr_file;

    DIR *dir = opendir(labels_dir_path.c_str());
    std::vector<std::string> filelists;
    unsigned short int jdx = 0;
    unsigned short int idx = 0;

    std::srand (unsigned(std::time(0)));

    // allocate space for 3 sub-vectors
    output.resize(3);

    // collecting all the files that contain labels
    if(dir == NULL){
        std::cout << "[ERROR] directory " << labels_dir_path << " doesn't exist. " << std::endl;
        exit(EXIT_FAILURE);
    }
    else{
        struct dirent *entry ;
        std::string labels_file_path;
        while ((entry = readdir(dir)) != NULL) 
        {
            if (entry->d_type == DT_REG) {
                labels_file_path = labels_dir_path + "/" + entry->d_name;
                filelists.push_back(labels_file_path);
            }
        }
        closedir( dir );
    }

    // collecting all labels from these files
    std::ifstream lblfs;

    for (std::vector<std::string>::iterator it = filelists.begin(); it != filelists.end(); ++ it )
    {
        lblfs.open(it->c_str(), std::ifstream::in);
        if(lblfs.is_open()) {
            char rawlinebuf [256];
            while (lblfs.getline(rawlinebuf, 256)) {
                std::istringstream fieldstream (rawlinebuf);
                std::string field;
                laneinfo laneinfonode ;
                // in each row we only need path of image example,
                // and the image coordinate indicating the centroid of the lane
                std::getline(fieldstream, field, ',');
                laneinfonode.imgpath = field;
                std::getline(fieldstream, field, ',');
                laneinfonode.label.x = NORMALIZE_NUMBER(std::stof(field), ORIGINAL_IMAGE_WIDTH, 0) ;
                std::getline(fieldstream, field, ',');
                laneinfonode.label.y = NORMALIZE_NUMBER(std::stof(field), ORIGINAL_IMAGE_HEIGHT, 0);
                laneinfonode.label.x = std::floor(1000 * laneinfonode.label.x) / 1000; 
                laneinfonode.label.y = std::floor(1000 * laneinfonode.label.y) / 1000; 
                laneinfonode.pred.x = 0;
                laneinfonode.pred.y = 0;
                labels_of_curr_file.push_back(laneinfonode);
            }
            lblfs.close();
        }
        std::vector<int> num_samples(3) ;
        num_samples[0] = labels_of_curr_file.size()*74/100-1;
        num_samples[1] = labels_of_curr_file.size()*25/100-1;
        num_samples[2] = labels_of_curr_file.size()*1/100-1;
        // start splitting each label files to 3 sudsets :
        // training / cross-validation / test set
        std::random_shuffle(labels_of_curr_file.begin(), labels_of_curr_file.end());

        // assign part of the labels to each of output
        for (jdx=0; jdx<num_samples.size(); jdx++) {
            for (idx = 0; idx < num_samples[jdx]; idx++) {
                output.at(jdx).push_back (labels_of_curr_file.back());
                labels_of_curr_file.pop_back();
            }
        }
        
        labels_of_curr_file.clear();
    }

    return output;
}


void dataset_handler::load_labeled_examples(
                         std::vector<laneinfo> &sliced_samples,
                         tensorflow::Tensor &sample_data, 
                         tensorflow::Tensor &label_data )
{
    // flattened pixels over all image examples of current batch.
    std::vector<float> flatten_pxl;
    std::vector<float> flatten_labels;
    std::vector<laneinfo>::iterator  it;
    for (it = sliced_samples.begin(); it != sliced_samples.end(); it++)
    {
        // since we always create matrix using cv::imread, which internally calls cv::create()
        // there is no need to check if the pixels of a image are continuously put in memory
        cv::Mat img = cv::imread(it->imgpath);
        cv::Mat norm_img;
        preprocess_one_image (img, norm_img);
        flatten_pxl.insert ( flatten_pxl.end(), (float*)norm_img.datastart, (float*)norm_img.dataend );
        img.release();
        norm_img.release();
        flatten_labels.push_back (it->label.x);
        flatten_labels.push_back (it->label.y);
    }
    std::copy_n( flatten_pxl.begin(),    flatten_pxl.size(),     sample_data.flat<float>().data() );
    std::copy_n( flatten_labels.begin(), flatten_labels.size(),   label_data.flat<float>().data() );
}




void  dataset_handler::load_unlabeled_example (cv::Mat& img_in,   tensorflow::Tensor &sample_data )
{
    // to-do: optimization
    std::vector<float> flatten_pxl;
    cv::Mat norm_img;
    int norm_img_size = 0;

    preprocess_one_image (img_in, norm_img);
    norm_img_size =  norm_img.total() * norm_img.elemSize() / sizeof(float);
    std::copy_n( (float*)norm_img.datastart, norm_img_size,  sample_data.flat<float>().data() );
    norm_img.release ();
}





void  dataset_handler::preprocess_one_image (cv::Mat& img_in, cv::Mat& img_out)
{
    cv::Mat continuous_img;
    cv::Mat resized_img;
    // check if the pixels of input matrix are continuous on memory (without gap)
    if (img_in.isContinuous()) {
        continuous_img = img_in;
    }
    else {
        // make the pixel continuous
        continuous_img = img_in.clone();
    }
    // any input image will be normalized to the range from -1 to 1
    // resize & normalize image
    cv::Size scaled_size = cv::Size(__hyparams_int["resized_img_width"], __hyparams_int["resized_img_height"]);
    cv::resize (continuous_img, resized_img, scaled_size);
    resized_img.convertTo (img_out, CV_32FC3, 2.0/255.0, -1.0);
    // append flattened pixels of current image to the vector of unsigned char
    resized_img.release();
    if (!img_in.isContinuous()) {
        continuous_img.release();
    }
}



