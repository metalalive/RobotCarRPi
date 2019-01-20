#include <vector> 
#include <queue>
#include <mutex>
#include <unistd.h>         // for usleep()
#include <pthread.h>

#include <opencv2/opencv.hpp>

#include <GPIOcontroller.h>
#include <dataset_handler.h>
#include <models.h>

// define a period of microsecond for operation of pulse width modulation,
// duty cycle represents from 0% to 100% of a PWM_PERIOD_US period, 
// which means the total amount of time PWM signal is asserted in that PWM_PERIOD_US period. 
#define PWM_PERIOD_US          200

// for each captured frame, there will be corresponding predicted centroid of lane,
// and generated duty cycles for both sides of DC motors,
// following parameter defines number of PWM_PERIOD_US time intervals to use for each captured frame.
#define NUM_PWM_PERIODS_FOR_EACH_FRAME  2

// global mutex shared among main (prediction) thread
// and polling thread (keep reading frames from camera)
std::mutex mtx;



void* polling_cam_frame (void* args)
{
    if (args==NULL){
        std::cout<< "[WARN] failed to pass parameters in polling_cam_frame"<< std::endl;
        return NULL;
    }
    std::vector<long int> readimg_thread_params = *(std::vector<long int> *) args;
    cv::VideoCapture     *cap          = (cv::VideoCapture *)    readimg_thread_params.at(0);
    std::queue<cv::Mat>  *framebuffer  = (std::queue<cv::Mat> *) readimg_thread_params.at(1);
    bool *pollframe_end_flag  = (bool*)       readimg_thread_params.at(2);
    // cv::Mat must be declared behind the VideoCapture
    // we haven't figured out why's that
    cv::Mat frame;
    cv::Mat rotated_180_frm;
    while (!(*pollframe_end_flag)) {
        usleep(35333);
        cap->read(frame);
        cv::rotate (frame, rotated_180_frm, cv::ROTATE_180);
        frame.release();
        mtx.lock();
        if (framebuffer->size() >= 1) {
            framebuffer->front().release();
            framebuffer->pop();
        }
        framebuffer->push(rotated_180_frm);
        mtx.unlock();
    } 
    std::cout<< "[INFO] end of polling_cam_frame "<< std::endl;
}



// convert points on 2D image to duty cycle of DC motor A & B,
// To control average voltage of these motors, you can produce differnt 
// duty cycle for each side of motors, then control how much to steer the car .
float pre_error_x = 0.0;

void point2D_2_pwm (std::vector<point2D>& lane_centroid_log, std::vector<int>& duty_cycle_AB )
{
    // give upper/lower bound of duty cycle for PWM signals of L298N
    const int MAX_DUTY_CYCLE_PWM = 90;
    const int MIN_DUTY_CYCLE_PWM = 5;

    float __dt  =  1.0 * PWM_PERIOD_US * NUM_PWM_PERIODS_FOR_EACH_FRAME;
    float kp_xa =  1.0 * (MAX_DUTY_CYCLE_PWM + MIN_DUTY_CYCLE_PWM) * 0.35 ;
    float kp_xb = -1.0 * (MAX_DUTY_CYCLE_PWM + MIN_DUTY_CYCLE_PWM) * 0.35 ;
    float kd_xa =  1.0 * (MAX_DUTY_CYCLE_PWM + MIN_DUTY_CYCLE_PWM) * 280.0 ;
    float kd_xb = -1.0 * (MAX_DUTY_CYCLE_PWM + MIN_DUTY_CYCLE_PWM) * 280.0 ;
    float kp_y  =        (MAX_DUTY_CYCLE_PWM + MIN_DUTY_CYCLE_PWM) * 0.47 ;
    
    float error_x  =  lane_centroid_log.back().x ;
    float error_y  = (lane_centroid_log.back().y * -1.0 + 1) / 2 + 0.05;
    float error_de_dt = (error_x - pre_error_x) / __dt ;

    float p_xa = error_x * kp_xa;
    float p_xb = error_x * kp_xb;
    float d_xa = error_de_dt * kd_xa;
    float d_xb = error_de_dt * kd_xb;
    float p_y  = error_y * kp_y;

    float pid_out_a = p_xa + p_y + d_xa ;
    float pid_out_b = p_xb + p_y + d_xb ;

    if (pid_out_a > MAX_DUTY_CYCLE_PWM)
        pid_out_a = MAX_DUTY_CYCLE_PWM;
    else if (pid_out_a < MIN_DUTY_CYCLE_PWM) 
        pid_out_a = MIN_DUTY_CYCLE_PWM;

    if (pid_out_b > MAX_DUTY_CYCLE_PWM)
        pid_out_b = MAX_DUTY_CYCLE_PWM;
    else if (pid_out_b < MIN_DUTY_CYCLE_PWM) 
        pid_out_b = MIN_DUTY_CYCLE_PWM;

    duty_cycle_AB.at(0) = pid_out_a;
    duty_cycle_AB.at(1) = pid_out_b;
    pre_error_x = error_x;
}




void drive_motors (std::vector<int>& duty_cycle_AB,
         GPIOcontroller& l298_in1,  GPIOcontroller& l298_in2,
         GPIOcontroller& l298_in3,  GPIOcontroller& l298_in4,
         GPIOcontroller& l298_enA,  GPIOcontroller& l298_enB
    )
{
    int idx;
    // to finely control pwm, we slice a PWM_PERIOD_US to 100 smaller time intervals,
    // in each time interval, 
    int pwm_toggle_period_us  = PWM_PERIOD_US / 100; 
    int num_times_chk = NUM_PWM_PERIODS_FOR_EACH_FRAME * 100 ;

    int duty_cyc_a = duty_cycle_AB.at(0);
    int duty_cyc_b = duty_cycle_AB.at(1);

    if(duty_cyc_a > 0) {
        l298_in1.set_value (1);
        l298_in2.set_value (0);
    }
    else if(duty_cyc_a < 0) {
        l298_in1.set_value (0);
        l298_in2.set_value (1);
    }

    if(duty_cyc_b > 0) {
        l298_in3.set_value (1);
        l298_in4.set_value (0);
    }
    else if(duty_cyc_b < 0) {
        l298_in3.set_value (0);
        l298_in4.set_value (1);
    }

    for (idx=0; idx < num_times_chk ; idx++)
    {
        if ((idx % 100) < duty_cyc_a ) { l298_enA.set_value (1); }
        else {   l298_enA.set_value (0);   }
        if ((idx % 100) < duty_cyc_b ) { l298_enB.set_value (1); }
        else {   l298_enB.set_value (0);   }
        usleep (pwm_toggle_period_us);
    }
    // you MUST clean the status of GPIO after each time driving the motors
    // the frequency of toggling GPIO pins seem limited & not that reliable
    l298_in1.set_value(0);
    l298_in2.set_value(0);
    l298_in3.set_value(0);
    l298_in4.set_value(0);
    l298_enA.set_value(0);
    l298_enB.set_value(0);
}




void append_pred_centroid_log (std::vector<point2D>& lane_centroid_log, point2D& curr_pred_lane_centroid)
{
    if (lane_centroid_log.size() >= 16) {
        lane_centroid_log.erase(lane_centroid_log.begin());
    }
    lane_centroid_log.push_back(curr_pred_lane_centroid);
}





void init_camera (cv::VideoCapture& cap, std::string img_src_path,
                  const int frame_width, const int frame_height)
{
    int idx = 0;
    cap.open (img_src_path);
    if (!cap.isOpened()) {
        std::cout<< "[ERROR] failed to open camera, please recheck the connection" << std::endl;
        exit(EXIT_FAILURE);
    }
    cap.set (cv::CAP_PROP_FRAME_WIDTH , frame_width  );
    cap.set (cv::CAP_PROP_FRAME_HEIGHT, frame_height );
    cap.set (cv::CAP_PROP_FPS, 30);
    cap.set (cv::CAP_PROP_BRIGHTNESS, 58);
    cap.set (cv::CAP_PROP_CONTRAST, 6);
    
    cv::Mat junkframes ;
    for (idx=0; idx<20 ; idx++) {
        cap >> junkframes ; 
        junkframes.release();
    }
}



void help_doc () {
    std::cout << "[ERROR] arguments missing " << std::endl;
    std::cout << " --------------------------------------------------------------------------------" << std::endl;
    std::cout << "     you should give following arguments: " << std::endl;
    std::cout << "     ./main.out  <CAMERA_PATH>   <TRAINED_MODEL_PATH>   <TRAINED_PARAM_PATH>   \\ " << std::endl;
    std::cout << "                 <NUM_FRAMES_FOR_TEST>    " << std::endl;
    std::cout << std::endl;
    std::cout << "     main.cc is the entry point to Machine-Learning-based autonomous toy car. " << std::endl;
    std::cout << "     which load neural network model saved in googld protobuf format. " << std::endl;
    std::cout << "     You'll need to provide trained model file & parameters. " << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
}




int  main (int argc, char ** argv) 
{
    if (argc < 4) {
        help_doc (); return 1;
    }
    std::string img_src_path = std::string(argv[1]) ;
    std::string model_path   = std::string(argv[2]) ;
    std::string params_path  = std::string(argv[3]) ;
    int         num_frames   = std::stoi  (argv[4]) ;

    const int  frame_width  = 640;
    const int  frame_height = 480;
    const short int  frame_channels = 3;
    const int  resized_img_width  = 40;
    const int  resized_img_height = 15;
    const int  RPI_GPIO_2_L298N_IN1 = 17;
    const int  RPI_GPIO_2_L298N_IN2 = 27;
    const int  RPI_GPIO_2_L298N_IN3 = 22;
    const int  RPI_GPIO_2_L298N_IN4 = 23;
    const int  RPI_GPIO_2_L298N_ENA = 3;
    const int  RPI_GPIO_2_L298N_ENB = 4;

    int idx     = 0;

    // based on each captured frame, once we get predicted lane centroid, 
    // we apply pulse width modulation to our car, make it move forward / steer to centain angle,
    // motor_duty_cycle[*] is used for controlling duty cycle represents  for both sides of DC motor,
    //     motor_duty_cycle[0] means duty cycle left-side  motor A
    //     motor_duty_cycle[1] means duty cycle right-side motor B
    std::vector<int> motor_duty_cycle (2);

    // log the 16 most recently predicted centroids, 
    // can also help prediction work better when camera couldn't capture lanes.
    std::vector<point2D> lane_centroid_log ;

    // we will use 2 threads to increase frame throughput
    //     one thread is polling camera (read-image thread),  
    //     the other thread is to predict lane using loaded neural network & captured frame (prediction thread)

    // prediction thread will singal this flag to indicate when the polling process finishes
    bool pollframe_end_flag  = false;

    // init camera
    cv::VideoCapture cap;
    init_camera (cap, img_src_path, frame_width, frame_height);

    // here we assume framebuffer is a FIFO queue of size 5
    std::queue<cv::Mat> framebuffer;

    // data & neural network model handling
    std::unordered_map< std::string, std::string>  hyparams_str;
    std::unordered_map< std::string, int>          hyparams_int;
    std::unordered_map< std::string, float>        hyparams_float;
    hyparams_str["model_path"]         = model_path  ;
    hyparams_str["trained_param_path"] = params_path ;
    hyparams_int["resized_img_width"]  = resized_img_width;
    hyparams_int["resized_img_height"] = resized_img_height;
    hyparams_int["num_samples_batch"]  = 1;
    hyparams_int["num_fc_inl"]  = resized_img_width * resized_img_height * frame_channels;
    hyparams_int["num_fc_outl"] = 2;

    dataset_handler dh (hyparams_int);
    models nnm {true, hyparams_str, hyparams_float, hyparams_int};
    nnm.initialize();

    // init GPIO pins
    GPIOcontroller l298in1 {std::to_string(RPI_GPIO_2_L298N_IN1)};
    GPIOcontroller l298in2 {std::to_string(RPI_GPIO_2_L298N_IN2)};
    GPIOcontroller l298in3 {std::to_string(RPI_GPIO_2_L298N_IN3)};
    GPIOcontroller l298in4 {std::to_string(RPI_GPIO_2_L298N_IN4)};
    GPIOcontroller l298enA {std::to_string(RPI_GPIO_2_L298N_ENA)};
    GPIOcontroller l298enB {std::to_string(RPI_GPIO_2_L298N_ENB)};
    l298in1.set_direction ("out");
    l298in2.set_direction ("out");
    l298in3.set_direction ("out");
    l298in4.set_direction ("out");
    l298enA.set_direction ("out");
    l298enB.set_direction ("out");

    // ready to create read-image thread
    pthread_t  readimg_thread_obj;
    std::vector<long int> readimg_thread_params ;
    readimg_thread_params.resize(4);
    readimg_thread_params.at(0) = (long int) (&cap) ;
    readimg_thread_params.at(1) = (long int) (&framebuffer) ;
    readimg_thread_params.at(2) = (long int) (&pollframe_end_flag);
    pthread_create (&readimg_thread_obj, NULL, polling_cam_frame, (void*)&readimg_thread_params);

    for (idx=0; idx<num_frames ; idx++) {
        while (framebuffer.size() == 0) {
            // keep polling until we capture new frame
            usleep(100);
        }
        
        // step 1, feed captured frame to neural network model
        point2D     curr_pred_lane_centroid ;
        mtx.lock();
        nnm.predict (framebuffer.front(), curr_pred_lane_centroid, dh);
        framebuffer.front().release();
        framebuffer.pop();
        mtx.unlock();
        append_pred_centroid_log (lane_centroid_log, curr_pred_lane_centroid);

        // step 2, convert predicted normalized  (x,y) value to duty cycle of both DC motors
        point2D_2_pwm( lane_centroid_log, motor_duty_cycle );

        // step 3, control motors
        drive_motors (motor_duty_cycle, l298in1,  l298in2,
                      l298in3,  l298in4,  l298enA,  l298enB);

    }

    pollframe_end_flag = true;
    pthread_join( readimg_thread_obj, NULL);
    cap.release ();
    l298in1.disable(); 
    l298in2.disable(); 
    l298in3.disable(); 
    l298in4.disable(); 
    l298enA.disable(); 
    l298enB.disable();
}

