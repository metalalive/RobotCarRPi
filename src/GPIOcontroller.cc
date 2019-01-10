#include "GPIOcontroller.h"


// 2 types of constructors available
// set GPIO 2 as default
GPIOcontroller::GPIOcontroller() : gpionum("2")
{
    enable();
}


// create GPIO class that controls given GPIO number num
GPIOcontroller::GPIOcontroller(std::string num) : gpionum(num)
{
    enable();
}

GPIOcontroller::~GPIOcontroller()
{
    disable();
}

void GPIOcontroller::enable()
{
    std::string filepath = "/sys/class/gpio/export";
    std::fstream  f (filepath.c_str(), std::ios_base::out);
    if (!f.is_open()) {
        std::cout << "[ERROR] cannot find "<< filepath <<", recheck your GPIO pinout on RPi."  << std::endl;
        exit(EXIT_FAILURE);
    }
    f << this->gpionum;
    f.close();
}



void GPIOcontroller::disable()
{
    std::string filepath = "/sys/class/gpio/unexport";
    std::fstream  f (filepath.c_str(), std::ios_base::out);
    if (!f.is_open()) {
        std::cout << "[ERROR] cannot find "<< filepath <<", recheck your GPIO pinout on RPi."  << std::endl;
        exit(EXIT_FAILURE);
    }
    f << this->gpionum;
    f.close();
}


// the direction of any GPIO should be:
// input, output, ot alternative functions 1-5
void GPIOcontroller::set_direction(const char* func)
{
    std::string filepath = "/sys/class/gpio/gpio"+ this->gpionum +"/direction";
    std::fstream  f (filepath.c_str(), std::ios_base::out);
    if (!f.is_open()) {
        std::cout << "[ERROR] cannot find "<< filepath <<", recheck your GPIO pinout on RPi."  << std::endl;
        return;
    }
    f << func;
    f.close();
}

void GPIOcontroller::set_value(int val)
{
    std::string filepath = "/sys/class/gpio/gpio"+ this->gpionum +"/value";
    std::fstream  f (filepath.c_str(), std::ios_base::out);
    if (!f.is_open()) {
        std::cout << "[ERROR] cannot find "<< filepath <<", recheck your GPIO pinout on RPi."  << std::endl;
        return;
    }
    f << val;
    f.close();
}

int  GPIOcontroller::get_value( )
{
    std::string filepath = "/sys/class/gpio/gpio"+ this->gpionum +"/value";
    std::fstream  f (filepath.c_str(), std::ios_base::in);
    int val = 0;
    if (!f.is_open()) {
        std::cout << "[ERROR] cannot find "<< filepath <<", recheck your GPIO pinout on RPi."  << std::endl;
        return -1;
    }
    f >> val;
    return val;
}

int  GPIOcontroller::get_gpio_num()
{
    return std::stoi(this->gpionum);
}



