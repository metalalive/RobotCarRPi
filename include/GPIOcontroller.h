#ifndef __GPIO_CTRLER_H
#define __GPIO_CTRLER_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>

class GPIOcontroller 
{

private:
    std::string gpionum;


public:
    // 2 types of constructors available
    GPIOcontroller();

    // create GPIO class that controls given GPIO number x
    GPIOcontroller(std::string num);

    ~GPIOcontroller ();

    void enable();

    void disable();

    // the direction of any GPIO should be:
    // input, output, ot alternative functions 1-5
    void set_direction(const char* func);

    void set_value(int val);

    int  get_value( );

    int  get_gpio_num();


};

#endif // end of  __GPIO_CTRLER_H

